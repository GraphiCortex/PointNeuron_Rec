"""Image-only review candidates and immutable human decisions for stage 1.

Review exports are diagnostic pilot inputs, not automatically promoted training
caches. References never choose image points or image skeletons.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

import numpy as np

from pointneuron.data.medial_sampling import MedialSamplingConfig, image_medial_skeleton
from pointneuron.data.training_cache import brightest_cell_representatives, indices_to_point_array


def spacing_metadata(path):
    if path is None:
        return {'status': 'missing', 'relative_xyz': None, 'physical_xyz_um': None}
    path = Path(path)
    raw = path.read_text(encoding='utf-8-sig').strip()
    match = re.fullmatch(r'\s*([0-9.eE+-]+)\s*x\s*([0-9.eE+-]+)\s*x\s*([0-9.eE+-]+)\s*(?:um|µm|μm)\s*', raw)
    if not match:
        return {'status': 'unrecognized', 'path': str(path), 'raw': raw, 'relative_xyz': None,
                'physical_xyz_um': None}
    physical = np.array([float(value) for value in match.groups()])
    if not np.isfinite(physical).all() or np.any(physical <= 0):
        raise ValueError('Invalid voxel spacing metadata')
    return {'status': 'available', 'path': str(path), 'raw': raw,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'physical_xyz_um': physical.tolist(), 'relative_xyz': (physical / physical[0]).tolist()}


def farthest_indices(xyz, count, spacing=(1., 1., 1.)):
    """Deterministic image-coordinate coverage; no reference input."""
    xyz = np.asarray(xyz, dtype=np.float64) * np.asarray(spacing)
    if count < 1 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Invalid bounded point selection')
    if len(xyz) <= count:
        return np.arange(len(xyz))
    selected = np.empty(count, dtype=np.int64)
    used = np.zeros(len(xyz), dtype=bool)
    distances = np.full(len(xyz), np.inf)
    current = int(np.argmax(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1)))
    for i in range(count):
        selected[i] = current
        used[current] = True
        distances = np.minimum(distances, np.sum((xyz - xyz[current]) ** 2, axis=1))
        distances[used] = -1
        current = int(np.argmax(distances))
    return np.sort(selected)


def review_settings(payload, base):
    allowed = {'threshold_fraction', 'spacing_xyz', 'teasar_scale', 'teasar_const',
               'dust_threshold', 'candidate_method', 'guide_method', 'cell_size'}
    if set(payload) - allowed:
        raise ValueError('Unknown review settings')
    threshold = float(payload['threshold_fraction'])
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError('Threshold fraction must be finite in [0,1)')
    if payload['candidate_method'] not in ('uniform', 'spatial') or payload['guide_method'] not in ('uniform', 'farthest'):
        raise ValueError('Unknown pool reduction method')
    cell = payload['cell_size']
    if isinstance(cell, bool) or not isinstance(cell, int) or not 1 <= cell <= 32:
        raise ValueError('Cell size must be an integer in [1,32]')
    medial = replace(base, **{key: payload[key] for key in
                     ('spacing_xyz', 'teasar_scale', 'teasar_const', 'dust_threshold')})
    # Review UI uses relative XYZ spacing: one X voxel is the distance unit.
    if abs(medial.spacing_xyz[0] - 1) > 1e-6 or max(medial.spacing_xyz) > 100:
        raise ValueError('Use relative spacing with X=1 and every axis <=100')
    return {**payload, 'threshold_fraction': threshold, 'spacing_xyz': list(medial.spacing_xyz)}, medial


def build_review_candidate(image, lower, settings, base, candidate_count=1024, guide_count=512,
                           seed=0, datatype=1, existing_skeleton=None):
    settings, medial = review_settings(settings, base)
    image = np.asarray(image)
    depth, height, width = image.shape
    threshold = round((255 if datatype == 1 else 65535) * settings['threshold_fraction'])
    data = image.ravel()
    foreground = np.flatnonzero(data > threshold)
    if not len(foreground):
        raise ValueError('No foreground survives: adjust extraction settings')
    rng = np.random.default_rng(seed)
    if len(foreground) >= candidate_count:
        uniform = np.sort(rng.choice(foreground, candidate_count, replace=False))
    else:
        uniform = np.sort(np.concatenate((foreground, rng.choice(foreground, candidate_count - len(foreground)))))
    if existing_skeleton is None:
        guide, edges = image_medial_skeleton(foreground, width, height, medial)
        guide += np.asarray(lower)
    else:
        guide, edges = existing_skeleton
    if settings['candidate_method'] == 'spatial' and len(foreground) >= candidate_count:
        representatives = brightest_cell_representatives(foreground, data, width, height, settings['cell_size'])
        xyz = indices_to_point_array(representatives, data, width, height)[:, :3]
        chosen = representatives[farthest_indices(xyz, candidate_count, medial.spacing_xyz)]
        if len(chosen) < candidate_count:
            remaining = np.setdiff1d(foreground, chosen)
            chosen = np.concatenate((chosen, rng.choice(remaining, candidate_count - len(chosen), replace=False)))
        candidates = np.sort(chosen)
    else:
        candidates = uniform
    if guide_count is not None and len(guide) > guide_count:
        selected = (farthest_indices(guide, guide_count, medial.spacing_xyz)
                    if settings['guide_method'] == 'farthest' else np.sort(rng.choice(len(guide), guide_count, replace=False)))
        bounded_guide = guide[selected]
    else:
        bounded_guide = guide.copy()
    points = indices_to_point_array(candidates, data, width, height)
    points[:, :3] += np.asarray(lower)
    return {'points': points, 'full_guide_xyz': guide, 'guide_edges': edges,
            'image_guide_xyz': bounded_guide, 'foreground_xyz':
            indices_to_point_array(foreground, data, width, height)[:, :3] + np.asarray(lower),
            'foreground_count': len(foreground), 'threshold': threshold,
            'settings': settings, 'medial_config': medial.to_dict()}


def write_review_decision(bundle, case, state, decision, reviewer, notes='', reference_exposed=False):
    if decision not in ('accepted', 'rejected', 'needs_adjustment') or not reviewer.strip():
        raise ValueError('Choose a decision and enter a reviewer name')
    if decision == 'accepted' and state['full_guide_vertices'] == 0:
        raise ValueError('Cannot accept an empty skeleton guide')
    if decision != 'accepted' and not notes.strip():
        raise ValueError('Record why this candidate needs adjustment or rejection')
    artifact = Path(bundle) / state['artifact']
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual != state['artifact_sha256']:
        raise ValueError('Review candidate changed: recompute before recording a decision')
    record = {'schema_version': 1, 'case_id': case['case_id'], 'sample_id': case['sample_id'],
              'partition': case['partition'], 'revision': state['revision'], 'decision': decision,
              'reviewer': reviewer.strip(), 'notes': notes.strip(),
              'reviewed_at': datetime.now(timezone.utc).isoformat(),
              'settings': state['settings'], 'artifact': state['artifact'], 'artifact_sha256': actual,
              'source_volume_sha256': case['source_volume_sha256'],
              'source_swc_sha256': case['source_swc_sha256'],
              'extraction_source_sha256': state.get('source_sha256', {}),
              'kimimaro_version': state.get('kimimaro_version'),
              'reference_overlay_exposed': bool(reference_exposed),
              'scope': 'internal-development manual review; not automatically promoted to training',
              'eligible_for_image_only_validation': case['partition'] != 'val' or not reference_exposed,
              'training_ready': False}
    output = Path(bundle) / 'reviews' / f"{case['case_id']}_r{state['revision']}_{uuid.uuid4().hex}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return output, record
