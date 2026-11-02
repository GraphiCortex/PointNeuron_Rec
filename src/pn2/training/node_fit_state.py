"""Immutable model/Adam/RNG/cursor checkpoints bound to one budget and dataset."""
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from pn2.inputs.node_cache import digest

SCHEMA='pn2-paired-node-fit-state-v1'


def capture(model,optimizer,bindings,step):
    if type(step) is not int or not 0<=step<=bindings['maximum_updates']:
        raise ValueError('Invalid update cursor')
    return {'schema':SCHEMA,'bindings':bindings,'step':step,'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),'model_training':model.training,
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
               'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}}


def restore(payload,model,optimizer,bindings):
    if (payload.get('schema')!=SCHEMA or payload.get('bindings')!=bindings
            or type(payload.get('step')) is not int or not 0<=payload['step']<=bindings['maximum_updates']):
        raise ValueError('Checkpoint budget/data/code bindings or update cursor differ')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
    model.train(payload['model_training']);rng=payload['rng']
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu'].cpu())
    if rng['torch_cuda'] is not None:torch.cuda.set_rng_state_all([r.cpu() for r in rng['torch_cuda']])
    return payload['step']


def save(path,model,optimizer,bindings,step):
    path=Path(path)
    with path.open('xb') as stream:
        torch.save(capture(model,optimizer,bindings,step),stream);stream.flush();os.fsync(stream.fileno())
    with path.with_suffix('.complete.json').open('x',encoding='utf-8') as stream:
        json.dump({'checkpoint':path.name,'sha256':digest(path),'step':step},stream,indent=2)
        stream.flush();os.fsync(stream.fileno())


def load(path,model,optimizer,bindings,*,map_location='cpu'):
    path=Path(path);marker=json.loads(path.with_suffix('.complete.json').read_text())
    if marker['checkpoint']!=path.name or marker['sha256']!=digest(path):
        raise ValueError('Checkpoint is incomplete or changed')
    payload=torch.load(path,map_location=map_location,weights_only=False)
    if payload['step']!=marker['step']:raise ValueError('Checkpoint marker/cursor mismatch')
    return restore(payload,model,optimizer,bindings)
