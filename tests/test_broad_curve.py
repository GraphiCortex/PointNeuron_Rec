from pathlib import Path
import importlib.util
import sys
import numpy as np

from pn2.evaluation.fixed_curve_graph import gate, build, score
from pn2.evaluation.formulation import segment_agreement


def summary(values):
    return {'neurons': [{'case_id': str(i), 'tolerances': {t: {'f1': float(x)} for t in ['2.0','6.0']}}
                        for i,x in enumerate(values)],
            'macro': {t: {'f1': float(np.mean(values))} for t in ['2.0','6.0']}}


def test_paired_gate_uses_strongest_control_and_requires_five_strict_wins():
    controls = {'weak':summary([.2]*7),'strong':summary([.5]*7)}
    result=gate(summary([.8]*4+[.49]*3),controls)
    assert not result['passed'] and result['checks']['2.0']['macro_gain_passed']
    assert result['checks']['2.0']['win_count']==4
    assert gate(summary([.65]*5+[.49]*2),controls)['passed']
    assert not gate(summary([.51]*7),controls)['passed']
    assert not result['training_gated']


def test_ablation_keeps_sato_sources_and_does_not_reapply_suppression_or_cutoff():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('broad_diag_fixture',scripts/'diagnose_broad_curve_controls.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=np.column_stack((np.arange(600)*3,np.zeros((600,2))))
    votes=np.zeros_like(source); support=np.zeros(600);sato=np.arange(600,dtype=float);ids=np.arange(600)
    arms, selected=module.ablation_sets(source,votes,support,sato,ids)
    assert len(arms['raw_votes_all'][0])==600
    assert len(arms['support512_no_offset'][0])==512  # even with all scores below .25
    assert len(arms['sato512_with_offset'][0])==512  # duplicates are not suppressed again
    np.testing.assert_array_equal(selected,np.arange(599,87,-1))
    np.testing.assert_array_equal(arms['sato512_with_offset'][1],selected)
    np.testing.assert_array_equal(arms['support512_no_offset'][0],source[:512])


def test_fixed_graph_matches_original_scorer_and_preserves_source_indices(tmp_path):
    p=np.array([[0.,1,1],[8.,1,1]])
    packet=build(p,np.ones((3,3,9)),1.,tmp_path/'graph',('case',1),'fixture',[7,11])
    reference=p[None];box={'start_xyz':[0,0,0],'stop_xyz':[9,3,3]}
    result=score(packet,reference,box)
    assert result['segments']==segment_agreement(reference,reference)
    with np.load(tmp_path/'graph/graph.npz') as a:np.testing.assert_array_equal(a['source_indices'],[7,11])
