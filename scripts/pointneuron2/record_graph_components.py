"""Record actual neuron-79 graph-component allocation without changing selection.

Prepare/check are CPU provenance checks. The user launches run separately:
reuse saved hybrid proposals, recompute the unchanged CPU geodesic graph once,
record pool matrices and actual selector calls, require the original graph again.
No learned inference, training, SWC export, repaired allocation or score tuning.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_ROOT/'src'))
sys.path.insert(0,str(REPO_ROOT))
import scripts.initialize_geodesic_graph as initializer
from scripts.pointneuron2.compare_hybrid_swc import verify_new
from scripts.pointneuron2.sampling_ablation import digest,read,write,live


class SelectionRecorder:
    """Observe existing function boundaries and return their original results."""

    def __init__(self, expected_pool):
        self.expected_pool=expected_pool
        self.original_adaptive=initializer.adaptive_connected_coverage_selection
        self.original_connected=initializer.connected_coverage_selection
        self.original_coverage=initializer.coverage_truncate
        self.pool=None
        self.adaptive_metadata=None
        self.selected=None
        self.component_trace=None
        self.active_calls=None

    @staticmethod
    def bound(function,args,kwargs):
        values=inspect.signature(function).bind(*args,**kwargs)
        values.apply_defaults()
        return values.arguments

    def adaptive(self,*args,**kwargs):
        if self.pool is not None:
            raise ValueError('Unexpected repeated adaptive-selector call')
        values=self.bound(self.original_adaptive,args,kwargs)
        for key in ['centers','radii','scores']:
            if not np.array_equal(values[key],self.expected_pool[key]):
                raise ValueError(f'Actual input pool differs from preparation: {key}')
        n=len(values['centers'])
        for key in ['geodesic','euclidean']:
            if values[key].shape!=(n,n) or np.isnan(values[key]).any():
                raise ValueError(f'Invalid pool matrix: {key}')
        self.pool={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in values.items()}
        print(f'Geodesics ready: {n} candidates. Recording the unchanged adaptive selector.',flush=True)
        selected,metadata=self.original_adaptive(*args,**kwargs)
        for key,value in self.pool.items():
            if isinstance(value,np.ndarray) and not np.array_equal(values[key],value):
                raise ValueError(f'Selector mutated an input array: {key}')
        self.selected=np.asarray(selected,dtype=np.int64).copy()
        self.adaptive_metadata=metadata.copy()
        return selected,metadata

    def connected(self,*args,**kwargs):
        if self.component_trace is not None or self.active_calls is not None:
            raise ValueError('Unexpected repeated/nested connected-selector call')
        values=self.bound(self.original_connected,args,kwargs)
        mask=initializer.geodesic_connectivity_mask(values['geodesic'],values['euclidean'],values['max_geodesic_ratio'])
        components=initializer.connected_components_from_mask(mask)
        components.sort(key=lambda c:(-c.size,-float(values['scores'][c].mean()) if c.size else 0.))
        eligible=[c for c in components if c.size>=3]
        if not eligible:
            eligible=components
        eligible_keys={tuple(c.tolist()) for c in eligible}
        self.active_calls=[]
        try:
            selected=self.original_connected(*args,**kwargs)
            calls=self.active_calls
        finally:
            self.active_calls=None
        records=[]
        for number,component in enumerate(components):
            component_calls=[c for c in calls if c['candidate_indices']==component.tolist() and not c['has_seeds']]
            indices=np.intersect1d(component,selected)
            records.append({'component_rank':number,'candidate_indices':component.tolist(),'candidate_count':int(component.size),
                'eligible':tuple(component.tolist()) in eligible_keys,
                'mean_score':float(values['scores'][component].mean()),
                'visited_in_allocation_loop':bool(component_calls),
                'requested_budgets':[c['requested_budget'] for c in component_calls],
                'selected_indices':indices.tolist(),'selected_count':len(indices)})
        self.component_trace={'components':records,'coverage_calls':calls,
            'max_nodes':int(values['max_nodes']),'max_geodesic_ratio':float(values['max_geodesic_ratio']),
            'selected_indices':np.asarray(selected,dtype=np.int64).tolist(),
            'eligible_components':len(eligible),
            'eligible_zero_anchor_components':[r['component_rank'] for r in records if r['eligible'] and r['selected_count']==0],
            'excluded_components':[r['component_rank'] for r in records if not r['eligible']]}
        return selected

    def coverage(self,*args,**kwargs):
        values=self.bound(self.original_coverage,args,kwargs)
        result=self.original_coverage(*args,**kwargs)
        if self.active_calls is not None:
            self.active_calls.append({'candidate_indices':values['candidate_indices'].tolist(),
                'requested_budget':int(values['max_nodes']),
                'has_seeds':values['seed_indices'] is not None,
                'selected_indices':np.asarray(result,dtype=np.int64).tolist()})
        return result

    @contextmanager
    def installed(self):
        initializer.adaptive_connected_coverage_selection=self.adaptive
        initializer.connected_coverage_selection=self.connected
        initializer.coverage_truncate=self.coverage
        try:
            yield self
        finally:
            initializer.adaptive_connected_coverage_selection=self.original_adaptive
            initializer.connected_coverage_selection=self.original_connected
            initializer.coverage_truncate=self.original_coverage


def verify_plan(plan):
    if digest(plan['comparison_path'])!=plan['comparison_sha256']:
        raise ValueError('Completed hybrid comparison changed')
    comparison=read(plan['comparison_path'])
    verify_new(comparison)
    for name,sha in plan['source_sha256'].items():
        if digest(REPO_ROOT/name)!=sha:
            raise ValueError(f'Trace source changed: {name}; prepare a new run')
    for key in ['proposal','expected_graph','expected_pool']:
        if digest(plan[key]['path'])!=plan[key]['sha256']:
            raise ValueError(f'Frozen {key} changed')
    return comparison


def prepare(comparison_path,output):
    if output.exists():
        raise FileExistsError('Use a fresh trace directory')
    comparison=read(comparison_path)
    if comparison['status']!='completed' or comparison.get('paired_checks')!='passed':
        raise ValueError('Requires the completed neuron-79 hybrid comparison')
    verify_new(comparison)
    arm=next(r for r in comparison['results'] if r['name']=='hybrid_50_50')
    proposal=Path(arm['reconstruction']['proposal_path']);graph=Path(arm['reconstruction']['graph_path'])
    if digest(proposal)!=arm['proposal_sha256'] or digest(graph)!=arm['graph_sha256']:
        raise ValueError('Saved hybrid proposal/graph changed')
    meta=arm['graph_metadata']
    with np.load(graph,allow_pickle=False) as g:
        if json.loads(str(g['metadata']))!=meta:
            raise ValueError('Saved graph metadata differs')
    with np.load(proposal,allow_pickle=False) as p:
        pool=initializer.filter_proposals(p['centers'],p['radii'],p['scores'],p['features'],
            min_score=meta['min_proposal_score'],nms_distance=meta['nms_distance'],nms_mode=meta['nms_mode'],
            iou_threshold=meta['iou_threshold'],max_nodes=meta['max_nodes'],selection_mode=meta['selection_mode'])
    if (len(pool[0])!=625 or meta['max_nodes']!=128 or meta['foreground_threshold']!=15
            or meta['min_proposal_score']!=.5 or meta['adaptive_selection']['selection_policy']!='connected_coverage'):
        raise ValueError('Unexpected scope/settings for this fixed trace')
    files=sorted((REPO_ROOT/'src').rglob('*.py'))+sorted((REPO_ROOT/'scripts').rglob('*.py'))
    files+=sorted((REPO_ROOT/'configs/pointneuron2').glob('*.json'))
    output.mkdir(parents=True)
    expected=output/'expected_pool.npz'
    np.savez_compressed(expected,**dict(zip(['centers','radii','scores','features','proposal_indices'],pool)))
    plan={'status':'prepared','scope':'one unchanged neuron-79 hybrid CPU graph replay; allocation recording only',
        'comparison_path':str(comparison_path),'comparison_sha256':digest(comparison_path),
        'proposal':{'path':str(proposal),'sha256':digest(proposal)},
        'expected_graph':{'path':str(graph),'sha256':digest(graph)},
        'expected_pool':{'path':str(expected.resolve()),'sha256':digest(expected)},
        'graph_metadata':meta,'source_sha256':{f.relative_to(REPO_ROOT).as_posix():digest(f) for f in files},
        'optimizer_steps':0,'new_learned_inference':False,'new_swc_export':False,'production_changed':False,
        'reserved_test_neurons_used':False,'device':'CPU (SciPy geodesics; no CUDA work)'}
    write(output/'trace_plan.json',plan)
    verify_plan(plan)
    print('Prepared: unchanged 625-candidate pool, cutoff=0.5, 128 anchors, threshold=15.',flush=True)
    print('No graph computation launched. User run repeats one CPU graph and records actual component allocation.',flush=True)
    print(f'Output directory: {output}',flush=True)


def initializer_arguments(plan,comparison,output):
    meta=plan['graph_metadata']
    args=['--root',comparison['baseline']['data_root'],'--sample-index','79',
        '--proposals',plan['proposal']['path'],'--output',str(output/'repeated_graph.npz')]
    mapping={'mode':'mode','knn':'knn','max_distance':'max_distance','min_proposal_score':'min_proposal_score',
        'nms_distance':'nms_distance','nms_mode':'nms_mode','iou_threshold':'iou_threshold','max_nodes':'max_nodes',
        'selection_mode':'selection_mode','foreground_threshold':'requested_foreground_threshold',
        'max_foreground_voxels':'max_foreground_voxels','intensity_penalty':'intensity_penalty',
        'disconnected_multiplier':'disconnected_multiplier','candidate_k':'candidate_k',
        'max_euclidean_distance':'max_euclidean_distance','max_geodesic_ratio':'max_geodesic_ratio',
        'bridge_max_geodesic_ratio':'bridge_max_geodesic_ratio'}
    for flag,key in mapping.items():
        args+=['--'+flag.replace('_','-'),str(meta[key])]
    for flag in ['allow_unreachable_fallback','bridge_components','bridge_allow_unreachable_fallback']:
        if meta[flag]:
            args+=['--'+flag.replace('_','-')]
    return args


@contextmanager
def progress_logging():
    originals={name:getattr(initializer,name) for name in ['read_volume','build_foreground_graph','dijkstra']}
    labels={'read_volume':'1/3: decode the frozen image on CPU',
        'build_foreground_graph':'2/3: build the unchanged foreground graph',
        'dijkstra':'3/3: compute paths from all 625 pool candidates (this is the main wait)'}
    def logged(name):
        def call(*args,**kwargs):
            start=time.perf_counter();print(labels[name],flush=True)
            result=originals[name](*args,**kwargs)
            print(f'{name} completed in {time.perf_counter()-start:.1f}s',flush=True)
            return result
        return call
    for name in originals:
        setattr(initializer,name,logged(name))
    try:
        yield
    finally:
        for name,function in originals.items():
            setattr(initializer,name,function)


def worker(output):
    plan=read(output/'trace_plan.json')
    if plan['status']!='running':
        raise ValueError('Worker must be invoked by the visible run stage')
    comparison=verify_plan(plan)
    with np.load(plan['expected_pool']['path'],allow_pickle=False) as pool:
        expected={k:pool[k].copy() for k in pool.files}
    recorder=SelectionRecorder(expected)
    original_argv=sys.argv
    try:
        sys.argv=[str(REPO_ROOT/'scripts/initialize_geodesic_graph.py')]+initializer_arguments(plan,comparison,output)
        with recorder.installed(),progress_logging():
            code=initializer.main()
        if code:
            raise ValueError(f'Graph initializer returned {code}')
    finally:
        sys.argv=original_argv
    if recorder.pool is None or recorder.component_trace is None or recorder.adaptive_metadata!=plan['graph_metadata']['adaptive_selection']:
        raise ValueError('Actual selector trace differs from the expected adaptive policy')
    with np.load(output/'repeated_graph.npz',allow_pickle=False) as actual,np.load(plan['expected_graph']['path'],allow_pickle=False) as old:
        if actual.files!=old.files or any(not np.array_equal(actual[k],old[k],equal_nan=True) for k in actual.files if k!='metadata'):
            raise ValueError('Repeated numerical graph differs from the saved hybrid graph')
        if json.loads(str(actual['metadata']))!=json.loads(str(old['metadata'])):
            raise ValueError('Repeated graph settings/metadata differ')
        if not np.array_equal(actual['original_indices'],expected['proposal_indices'][recorder.selected]):
            raise ValueError('Recorded selected indices do not match the repeated graph')
    arrays={k:v for k,v in recorder.pool.items() if isinstance(v,np.ndarray)}
    arrays.update(proposal_indices=expected['proposal_indices'],selected_pool_indices=recorder.selected)
    artifact=output/'pool_geodesics.npz';np.savez_compressed(artifact,**arrays)
    result={'status':'completed','observed_actual_neuron':79,'same_numeric_graph':'passed',
        'adaptive_metadata':recorder.adaptive_metadata,'component_trace':recorder.component_trace,
        'matrix_artifact':{'path':str(artifact.resolve()),'sha256':digest(artifact)},
        'repeated_graph':{'path':str((output/'repeated_graph.npz').resolve()),'sha256':digest(output/'repeated_graph.npz')},
        'comparison_sha256':plan['comparison_sha256'],'production_changed':False,'repair_implemented':False}
    write(output/'component_trace.json',result)
    for r in recorder.component_trace['components']:
        print(f"component={r['component_rank']} candidates={r['candidate_count']} eligible={r['eligible']} visited={r['visited_in_allocation_loop']} requested={r['requested_budgets']} selected={r['selected_count']}",flush=True)
    print('Eligible components with zero anchors:',recorder.component_trace['eligible_zero_anchor_components'],flush=True)
    print('Repeated graph: exactly identical numerical arrays and settings; recording did not change selection.',flush=True)


def run(output):
    path=output/'trace_plan.json';plan=read(path)
    if plan['status']!='prepared':
        raise ValueError('Preserve earlier/partial traces; prepare a fresh directory')
    verify_plan(plan)
    if any((output/name).exists() for name in ['repeated_graph.npz','component_trace.json','pool_geodesics.npz']):
        raise FileExistsError('Trace artifacts already exist')
    start=time.perf_counter();plan.update(status='running');write(path,plan)
    try:
        print('CPU graph recording only: same saved proposals and settings. No model inference/training/SWC export.',flush=True)
        live([sys.executable,'-u',str(Path(__file__)),'--stage','worker','--output-dir',str(output)],output/'console.log')
        verify_plan(plan)
        result=read(output/'component_trace.json')
        if result['status']!='completed' or result['same_numeric_graph']!='passed':
            raise ValueError('Graph recording did not complete its checks')
        for key in ['matrix_artifact','repeated_graph']:
            if digest(result[key]['path'])!=result[key]['sha256']:
                raise ValueError('Recorded trace artifact changed')
        plan.update(status='completed',recording_checks='passed',component_trace_sha256=digest(output/'component_trace.json'),
            eligible_zero_anchor_components=result['component_trace']['eligible_zero_anchor_components'],
            elapsed_seconds=time.perf_counter()-start)
    except BaseException as exc:
        plan.update(status='failed_or_interrupted',error=str(exc),elapsed_seconds=time.perf_counter()-start)
        raise
    finally:
        write(path,plan)
    print(f"Status: completed; recording checks: passed; elapsed={plan['elapsed_seconds']:.1f}s",flush=True)
    print(f'Results: {output / "component_trace.json"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','check','run','worker'],required=True)
    parser.add_argument('--comparison',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();output=args.output_dir.resolve()
    if args.stage=='prepare':
        if args.comparison is None:
            parser.error('prepare requires --comparison')
        prepare(args.comparison.resolve(),output)
    elif args.stage=='check':
        verify_plan(read(output/'trace_plan.json'));print('Frozen trace sources/inputs: passed; no graph calculation launched.')
    elif args.stage=='worker':
        worker(output)
    else:
        run(output)


if __name__=='__main__':
    main()
