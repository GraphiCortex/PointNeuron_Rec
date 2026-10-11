"""Versioned component allocation repair; production defaults are unchanged.

Keep component eligibility/order, Euclidean within-component coverage and the
existing global fill. Use the remaining budget in proportional shares, reserving
one anchor for each later eligible component when the total budget permits.
"""
from contextlib import contextmanager

import numpy as np
import scripts.initialize_geodesic_graph as initializer

VERSION = 'remaining_budget_reserved_v1'


def repaired_connected_coverage_selection(centers, scores, geodesic, euclidean, max_nodes, max_geodesic_ratio):
    if max_nodes <= 0 or len(centers) <= max_nodes:
        return np.arange(len(centers), dtype=np.int64)
    mask = initializer.geodesic_connectivity_mask(geodesic, euclidean, max_geodesic_ratio)
    components = initializer.connected_components_from_mask(mask)
    components.sort(key=lambda c: (-c.size, -float(scores[c].mean()) if c.size else 0.))
    eligible = [c for c in components if c.size >= 3] or components
    remaining_source = sum(c.size for c in eligible)
    guarantee_each = max_nodes >= len(eligible)
    selected = []
    selected_set = set()
    for position, component in enumerate(eligible):
        remaining = int(max_nodes) - len(selected)
        if remaining <= 0:
            break
        later_components = len(eligible) - position - 1
        available = remaining - later_components if guarantee_each else remaining
        budget = round(remaining * int(component.size) / remaining_source)
        budget = max(1, min(int(component.size), int(budget), available))
        picked = initializer.coverage_truncate(centers=centers, scores=scores,
            candidate_indices=component.astype(np.int64, copy=False), max_nodes=budget)
        for index in picked.tolist():
            if int(index) not in selected_set:
                selected.append(int(index))
                selected_set.add(int(index))
        remaining_source -= int(component.size)
    if len(selected) < int(max_nodes):
        seeds = np.asarray(selected, dtype=np.int64) if selected else None
        filled = initializer.coverage_truncate(centers=centers, scores=scores,
            candidate_indices=np.arange(len(centers), dtype=np.int64), max_nodes=int(max_nodes), seed_indices=seeds)
        selected = filled.tolist()
    return np.asarray(selected[:int(max_nodes)], dtype=np.int64)


@contextmanager
def installed():
    """Apply this one allocator only inside a diagnostic worker; restore on exit."""
    original = initializer.connected_coverage_selection
    initializer.connected_coverage_selection = repaired_connected_coverage_selection
    try:
        yield
    finally:
        initializer.connected_coverage_selection = original
