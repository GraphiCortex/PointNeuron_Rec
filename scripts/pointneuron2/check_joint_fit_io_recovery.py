"""Read-only CPU audit of the failed fit and its last complete recovery checkpoint."""
from pathlib import Path
import copy
import json
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_saved_adam_fit as fit


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def run():
    torch.set_num_threads(2)
    base=fit.runner.base;root=ROOT/'tmp/pointneuron2/local_runs/joint_saved_adam_fit_20261020'
    failed=root/'model';report=base.read(failed/'report.json');active=base.read(failed/'active_update.json')
    pending=base.read(failed/'active_update.json.writing')
    updates=[json.loads(line) for line in (failed/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert report['status']=='failed' and report['completed_updates']==148 and report['attempted_update']==149
    assert [u['stage_update'] for u in updates]==list(range(1,149))
    assert active['stage_update']==148 and active['optimizer_step_returned']
    assert pending=={'stage_update':149,'crop':[0,0],'optimizer_step_returned':False,'phase':'forward'}
    path=failed/'step_0008.pt';digest=base.file_digest(path)
    assert digest==report['last_checkpoint_sha256'] and report['last_checkpoint_step']==8
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['schema']==fit.SCHEMA and saved['stage_step']==8 and saved['next_crop']==0
    assert saved['purpose']=='fit' and saved['resumable'] and not saved['prefix_verified']
    config=base.read(fit.CONFIG);fit.validate_config(config);assert saved['config']==config
    recipe,_,_,_,protected,_,_=fit.load_inputs(config);assert saved['training_recipe']==recipe
    current=copy.deepcopy(saved['provenance']);controller=str(Path(fit.__file__).resolve())
    current['sources'][controller]=base.file_digest(Path(controller))
    transition=fit.source_transition(saved['provenance'],current)
    base.verify_hashes(current['sources'])
    model=fit.runner.warm_start(recipe,'cpu')
    assert saved['parameter_names']==[n for n,_ in model.named_parameters()]
    assert saved['proposal_spec']==model.proposal_model.model_spec() and saved['graph_spec']==model.graph_model.model_spec()
    model.load_state_dict(saved['model']);optimizer=fit.runner.optimizer_for(model,recipe)
    optimizer.load_state_dict(saved['optimizer']);fit.validate_optimizer(model,optimizer,8,recipe)
    assert same(saved['model'],model.state_dict()) and same(saved['optimizer'],optimizer.state_dict())
    for state in [saved['rng']['torch_cpu'],*saved['rng']['torch_cuda']]:
        assert state.device.type=='cpu' and state.dtype==torch.uint8 and state.numel()>0
    # CUDA RNG is preserved, never restored/initialized by this CPU audit.
    assert not torch.cuda.is_initialized()
    result={'status':'passed','scope':'CPU checkpoint audit; zero forwards, optimizer updates or CUDA RNG restoration',
        'completed_updates':148,'failed_before_forward':149,'recoverable_stage':8,'repeated_discarded_updates':140,
        'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':digest,'failed_report_sha256':base.file_digest(failed/'report.json'),
        'active_sha256':base.file_digest(failed/'active_update.json'),'pending_sha256':base.file_digest(failed/'active_update.json.writing'),
        'journal_sha256':base.file_digest(failed/'updates.jsonl'),'source_transition':transition,
        'cursors':fit.timing.stability.cursors(model,optimizer),'weights_and_optimizer_identical':True,
        'prefix_reverification_required':True,'protected_files_verified':len(protected),'cuda_initialized':False}
    output=root/'io_recovery_audit/summary.json'
    if output.exists():raise FileExistsError('Preserve prior audit')
    fit.write(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
