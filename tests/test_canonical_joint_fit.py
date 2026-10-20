"""Prospective geometry policy, full optimizer/RNG recovery and timing replay checks."""
import copy
import random

import numpy as np
import pytest
import torch

from scripts.pointneuron2 import canonical_joint_fit as fit
from test_joint_partial_stability import Model, initialized, RECIPE


def config():
    return fit.base.read(fit.CONFIG)


def geometry(length, bad, missed=0., reference=100.):
    return {'export_length': length, 'reference_length': reference,
        'tolerances': {'2.0': {'precision': 1 - bad / length if length else None, 'recall': 1 - missed / reference}}}


def test_shorter_curve_with_same_bad_length_is_not_new_geometric_error():
    assert fit.geometry_noninferior(geometry(999., .5), geometry(1000., .5), config())
    assert not fit.geometry_noninferior(geometry(999., .6), geometry(1000., .5), config())
    assert not fit.geometry_noninferior(geometry(100., 2.), geometry(200., 2.), config())
    assert not fit.geometry_noninferior(geometry(100., 0., missed=1.), geometry(100., 0.), config())
    assert not fit.geometry_noninferior(geometry(0., 0.), geometry(100., 0.), config())


def test_new_policy_keeps_structural_and_bound_requirements():
    keys = ('structural_topology_all_crops', 'no_new_candidate_losses', 'four_crop_candidate_recall_complete', 'all_targets_fit_capacity', 'exact_input_clouds_preserved')
    preparation = {'preparation_gate': dict.fromkeys(keys, True), 'records': [
        {'sample_index': c[0], 'patch_index': c[1], 'canonical_geometry': geometry(999., .5),
         'legacy_geometry': geometry(1000., .5), 'certificates': [{'maximum_curve_deviation': .2, 'maximum_radius_deviation': .2}]}
        for c in config()['training_crops']]}
    preparation['preparation_gate']['passed'] = False
    assert fit.eligibility(preparation, config())['passed']
    assert preparation['preparation_gate']['passed'] is False
    preparation['records'][0]['certificates'][0]['maximum_curve_deviation'] = .6
    assert not fit.eligibility(preparation, config())['passed']
    preparation['records'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        fit.eligibility(preparation, config())


def start(stage=0, graph=None):
    model, optimizer = initialized(); graph = stage if graph is None else graph
    for group in optimizer.param_groups:
        group['lr'] = 1e-5
    for p in model.proposal_model.parameters():
        optimizer.state[p]['step'].fill_(5000 + stage)
    for p in model.graph_model.parameters():
        optimizer.state[p]['step'].fill_(1000 + graph)
    return model, optimizer


@pytest.mark.parametrize('stage,graph', [(0, 0), (8, 8), (500, 499)])
def test_full_state_resume_and_next_update_are_exact(tmp_path, stage, graph):
    model, optimizer = start(stage, graph)
    report = {'proposal_updates': stage, 'graph_updates': graph, 'timing_prefix_verified': stage >= 8}
    saved = fit.checkpoint(model, optimizer, report, config(), RECIPE, {'backend': 'test'}, 'fit')
    path = tmp_path / 'state.pt'; torch.save(saved, path)
    x = torch.randn(2, 3); draws = (np.random.rand(), random.random())
    def update(m, o):
        o.zero_grad(set_to_none=True); proposal, edge = m.losses(x)
        fit.fit.apply_update(m, o, proposal + edge, True, {}, lambda: None, dict(RECIPE, gradient_clip=10.))
    update(model, optimizer)
    other = Model()
    loaded, q, g = fit.restore(torch.load(path, weights_only=True), other, config(), RECIPE, {'backend': 'test'}, resume=True)
    assert (q, g) == (stage, graph)
    assert torch.equal(x, torch.randn(2, 3)) and draws == (np.random.rand(), random.random())
    update(other, loaded)
    assert fit.control.equal(model.state_dict(), other.state_dict()) and fit.control.equal(optimizer.state_dict(), loaded.state_dict())


@pytest.mark.parametrize('purpose,stage', [('check', 0), ('timing', 8), ('geometry_stop', 64), ('fit', 3000)])
def test_nonresumable_purposes_and_terminal_cap(purpose, stage):
    model, optimizer = start(stage)
    saved = fit.checkpoint(model, optimizer, {'proposal_updates': stage, 'graph_updates': stage, 'timing_prefix_verified': stage >= 8}, config(), RECIPE, {}, purpose)
    assert not saved['resumable']
    with pytest.raises(ValueError, match='cannot resume'):
        fit.restore(saved, model, config(), RECIPE, {}, resume=True)
    fit.restore(saved, model, config(), RECIPE, {}, resume=False)


def test_fingerprint_covers_model_moments_and_rng():
    model, optimizer = start()
    before = fit.fingerprint(model, optimizer)
    state = copy.deepcopy(optimizer.state_dict())
    optimizer.state[next(model.parameters())]['exp_avg'].add_(1)
    assert fit.fingerprint(model, optimizer) != before
    optimizer.load_state_dict(state)
    assert fit.fingerprint(model, optimizer) == before
    torch.rand(1)
    assert fit.fingerprint(model, optimizer) != before


def test_config_and_checkpoint_source_clock_options_are_strict():
    changed = config(); changed['updates'] = 4000
    with pytest.raises(ValueError):
        fit.validate_config(changed)
    changed = config(); changed['baseline_report_sha256'] = 'bad'
    with pytest.raises(ValueError):
        fit.validate_config(changed)
    model, optimizer = start()
    report = {'proposal_updates': 0, 'graph_updates': 0, 'timing_prefix_verified': False}
    saved = fit.checkpoint(model, optimizer, report, config(), RECIPE, {}, 'fit')
    for mutate in (lambda s: s.update(next_crop=1), lambda s: s.update(timing_prefix_verified=True),
                   lambda s: s.update(historical_stage=2000), lambda s: s['optimizer']['param_groups'][0].update(lr=.001)):
        bad = copy.deepcopy(saved); mutate(bad)
        with pytest.raises(ValueError):
            fit.restore(bad, Model(), config(), RECIPE, {}, resume=True)


def test_initial_prediction_replay_compares_inference_not_changed_assignments(monkeypatch, tmp_path):
    fields = ('centers', 'radii', 'logits', 'decoder_features', 'candidate_edges', 'edge_logits', 'retained_node_ids', 'retained_edges', 'retained_edge_logits')
    reference = {k: np.zeros((2,)) for k in fields}
    reference['pred_indices'] = np.array([0])
    actual = {k: v.copy() for k, v in reference.items()}; actual['pred_indices'] = np.array([1])
    monkeypatch.setattr(fit.control, 'read_arrays', lambda path: actual if path.parent == tmp_path else reference)
    fit.verify_initial_prediction(tmp_path, config())
    actual['edge_logits'][0] = .1
    with pytest.raises(ValueError, match='initial prediction'):
        fit.verify_initial_prediction(tmp_path, config())
