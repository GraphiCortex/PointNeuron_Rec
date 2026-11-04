"""Continuation validation and cursor scheduling; no model/recipe changes."""
import numpy as np


def continuation_steps(start, maximum, boxes):
    if (any(type(v) is not int for v in (start, maximum, boxes))
            or start < 0 or start >= maximum or boxes <= 0):
        raise ValueError('Invalid continuation cursor, cap or box count')
    return [(step, (step-1) % boxes) for step in range(start+1, maximum+1)]


def prediction_replay(points, scores, saved_points, saved_scores):
    if (not np.array_equal(points, saved_points) or not np.array_equal(scores, saved_scores)
            or not np.isfinite(points).all() or not np.isfinite(scores).all()):
        raise ValueError('Restored predictions differ from saved update-1000 outputs; stop before training')


def metric_replay(step, measured, original):
    expected = {name: original[name] for name in measured}
    if measured != expected:
        raise ValueError(f'Replayed optimizer update {step} differs from the original trace; no continuation')


def validate_payload(payload, bindings, start, optimizer_settings):
    import torch
    from pn2.training.node_fit_state import SCHEMA
    if (payload.get('schema') != SCHEMA or payload.get('bindings') != bindings
            or payload.get('step') != start):
        raise ValueError('Recovery checkpoint identity/cursor mismatch')
    if not payload['model'] or any(not torch.isfinite(v).all() for v in payload['model'].values()):
        raise ValueError('Missing/nonfinite model state')
    groups = payload['optimizer']['param_groups']
    states = payload['optimizer']['state']
    ids = [p for group in groups for p in group['params']]
    if len(groups) != 1 or set(ids) != set(states) or len(ids) != len(set(ids)):
        raise ValueError('Incomplete or duplicated Adam parameter state')
    for value in states.values():
        if (float(value['step']) != start or not torch.isfinite(value['exp_avg']).all()
                or not torch.isfinite(value['exp_avg_sq']).all()):
            raise ValueError('Invalid Adam clock or moments')
    group = groups[0]
    for key, expected in (('lr', optimizer_settings['learning_rate']),
                          ('weight_decay', optimizer_settings['weight_decay']),
                          ('eps', optimizer_settings['eps']), ('betas', tuple(optimizer_settings['betas'])),
                          ('foreach', False), ('fused', False)):
        if group[key] != expected:
            raise ValueError(f'Adam setting changed: {key}')
    rng = payload['rng']
    if set(rng) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
        raise ValueError('Incomplete RNG state')
    return len(states)
