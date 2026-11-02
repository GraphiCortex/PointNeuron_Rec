import numpy as np
from pn2.evaluation.node_profile_metrics import score_profiles


def test_stratum_recall_keeps_unresolved_and_fine_curve_support_baseline():
    target=np.array([[0.,0.,0.],[8.,0.,0.]])
    fine=np.column_stack((np.arange(.25,8.,.5),np.zeros((16,2))))
    p={'node_width':np.array([1.,0.]),'node_width_valid':np.array([True,False]),
       'node_contrast':np.array([.1,0.]),'node_contrast_valid':np.array([True,False]),
       'window_width':np.array([1.,1.]),'window_width_valid':np.ones(2,bool),
       'window_contrast':np.array([.1,.1]),'window_contrast_valid':np.ones(2,bool),
       'curve_xyz':fine,'curve_weights':np.full(16,.5),'curve_window_index':np.r_[np.zeros(8,int),np.ones(8,int)]}
    r=score_profiles(target,np.ones(2),{'target_xyz':target},p,{'width_q25':2.,'contrast_q25':.2})['2.0']
    assert r['apparent_thin_dim']['reference_nodes']==1 and r['unresolved']['reference_nodes']==1
    assert sum(x['matched_nodes'] for x in r.values())==2
    assert r['apparent_thin_dim']['covered_curve_length']==4.
    assert r['apparent_thin_dim']['target_grid_covered_curve_length']==4.
