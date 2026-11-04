"""Ensure failed scientific gates cannot silently become advancement decisions."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1]/'scripts/local_curve_experiment.py'
    spec = importlib.util.spec_from_file_location('local_curve_gate_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_gate_keeps_bad_box_in_denominator(experiment, monkeypatch):
    c = {'rungs': {'4': {'train_precision': .9, 'train_coverage': .8,
                         'oracle_ratio': .9, 'minimum_box_oracle_ratio': .8}}}
    oracle = {'summaries': {'four': {'oracle': {'macro': {'coverage_2': 1.}}}},
              'rows': {'oracle': [{'case_id': 'a', 'tile_id': i, 'metrics': {'coverage_2': 1.}}
                                  for i in range(4)]}}
    monkeypatch.setattr(experiment, 'read', lambda _: oracle)
    terminal = {'summary': {'macro': {'precision_2': .99, 'coverage_2': .95}},
                'rows': [{'case_id': 'a', 'tile_id': i, 'metrics': {'coverage_2': 1. if i else .5}}
                         for i in range(4)]}
    gate = experiment.four_gate(c, terminal)
    assert not gate['passed']
    assert gate['checks']['macro_oracle_ratio']
    assert not gate['checks']['minimum_box_oracle_ratio']


@pytest.mark.parametrize('precision,coverage,worst,passes', [(.9, .8, .3, True),
    (.59, .8, .3, False), (.9, .59, .3, False), (.9, .8, .24, False)])
def test_development_absolute_floors_are_conjunctive(experiment, precision, coverage, worst, passes):
    c = {'rungs': {'16': {'train_precision': .85, 'train_coverage': .8,
        'development_precision': .6, 'development_coverage': .6,
        'minimum_development_neuron_coverage': .25}}}
    training = {'summary': {'macro': {'precision_2': .9, 'coverage_2': .9}}}
    dev = {'summary': {'macro': {'precision_2': precision, 'coverage_2': coverage},
                       'neurons': [{'coverage_2': worst}]}}
    assert experiment.absolute_sixteen_gate(c, training, dev)['passed'] is passes


def test_runner_rejects_64_before_loading_data(experiment, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['local_curve_experiment.py', 'train', '--rung', '64'])
    monkeypatch.setattr(experiment, 'control', lambda: pytest.fail('Must reject unauthorized rung first'))
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2
