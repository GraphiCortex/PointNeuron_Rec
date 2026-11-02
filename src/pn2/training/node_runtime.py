"""Model tensors without target access on the inference path."""
import numpy as np
import torch


def image_batch(record,device):
    packet=record['packet'];meta=record['metadata']
    tensor=lambda a:torch.from_numpy(np.asarray(a,dtype=np.float32).copy()).to(device)
    return {'local_xyz':tensor(packet['local_xyz'])[None],
        'attributes':tensor(packet['attributes'])[None],
        'points':tensor(np.column_stack((packet['native_xyz'],packet['raw_intensity'])))[None],
        'box_center':tensor(meta['center_xyz'])[None]}


def training_batch(record,device):
    if record['metadata']['partition']!='training' or not len(record['target_xyz']):
        raise ValueError('Development and unknown targets cannot enter gradient training')
    return {**image_batch(record,device),'target':torch.from_numpy(
        np.asarray(record['target_xyz'],dtype=np.float32).copy()).to(device)}
