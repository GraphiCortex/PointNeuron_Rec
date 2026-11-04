import numpy as np
import pytest

from pn2.training.local_curve_recovery import continuation_steps, prediction_replay, metric_replay, validate_payload


def test_continuation_uses_original_update_phase_and_cap():
    steps = continuation_steps(1000, 2000, 4)
    assert len(steps) == 1000
    assert steps[:4] == [(1001, 0), (1002, 1), (1003, 2), (1004, 3)]
    assert steps[-1] == (2000, 3)
    assert continuation_steps(1001, 1003, 4) == [(1002, 1), (1003, 2)]


@pytest.mark.parametrize('args', [(2000, 2000, 4), (-1, 2000, 4), (1000, 2000, 0), (True, 2000, 4)])
def test_invalid_or_completed_cursor_cannot_restart(args):
    with pytest.raises(ValueError):
        continuation_steps(*args)


def test_prediction_replay_requires_exact_positions_and_scores():
    p, s = np.zeros((3, 3)), np.ones(3)*.25
    prediction_replay(p, s, p.copy(), s.copy())
    changed = p.copy()
    changed[0, 0] = 1e-8
    with pytest.raises(ValueError):
        prediction_replay(changed, s, p, s)
    with pytest.raises(ValueError):
        prediction_replay(p, s+.01, p, s)


def test_optimizer_replay_cannot_hide_changed_loss():
    metric_replay(1001, {'loss': .5, 'gradient_norm': 1.}, {'step': 1001, 'loss': .5, 'gradient_norm': 1.})
    with pytest.raises(ValueError):
        metric_replay(1001, {'loss': .51}, {'step': 1001, 'loss': .5})


def test_checkpoint_validation_rejects_reset_adam_or_recipe():
    torch = pytest.importorskip('torch')
    from pn2.training.node_fit_state import capture
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=0., foreach=False, fused=False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    bindings = {'maximum_updates': 2, 'data': 'fixture'}
    settings = {'learning_rate': .0001, 'weight_decay': 0., 'eps': 1e-8, 'betas': [.9, .999]}
    payload = capture(model, optimizer, bindings, 1)
    assert validate_payload(payload, bindings, 1, settings) == 2
    with pytest.raises(ValueError):
        validate_payload(payload, bindings, 0, settings)
    with pytest.raises(ValueError):
        validate_payload(payload, bindings, 1, {**settings, 'learning_rate': .001})
    payload['optimizer']['state'].clear()
    with pytest.raises(ValueError):
        validate_payload(payload, bindings, 1, settings)
