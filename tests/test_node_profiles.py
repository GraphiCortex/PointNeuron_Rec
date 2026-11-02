import numpy as np
import pytest
from pn2.evaluation.node_profiles import weighted_cutoff,classify,build_profiles


def test_quantiles_use_training_arclength_weights_and_missing_values_stay_unresolved():
    assert weighted_cutoff(np.array([1.,9.]),np.array([3.,1.]),np.array([True,True]))==1.
    assert weighted_cutoff(np.array([1.]),np.array([1.]),np.array([False])) is None
    labels=classify(np.array([1.,10.,1.]),np.array([True,True,False]),np.array([0.,1.,0.]),
        np.ones(3,dtype=bool),{'width_q25':2.,'contrast_q25':.5})
    assert labels.tolist()==[1,4,0]


def test_fine_curve_samples_prevent_four_voxel_profile_aliasing():
    image=np.zeros((20,20,20),np.uint8);points=np.array([[4.,10.,10.],[12.,10.,10.]])
    reference={'target_xyz':points,'reference_runs':[points],
        'target_packet':{'identities_json':np.array(['["node",1]','["node",2]'])}}
    p=build_profiles(image,reference,{'start_xyz':[0,0,0],'stop_xyz':[20,20,20]},
        {'minimum':0.,'maximum':0.},lambda identity:np.array([1.,0.,0.]))
    assert len(p['window_weights'])==2 and len(p['curve_weights'])==16
    assert p['curve_weights'].sum()==8. and set(p['curve_window_index'])=={0,1}
    nearest=np.minimum(p['curve_xyz'][:,0]-4.,12.-p['curve_xyz'][:,0])
    assert p['curve_weights'][nearest<=2].sum()/8==.5
    assert not p['window_width_valid'].any() and not p['window_contrast_valid'].any()
