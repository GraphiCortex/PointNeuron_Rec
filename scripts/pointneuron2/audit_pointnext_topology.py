"""Compare pinned reference layer wiring with the native prototype on CPU.

Run from the PN2 directory. The ignored reference snapshot must already exist;
URLs and hashes are in docs/pointneuron2/pointnext_reference/reference_manifest.json.
This uses native point operations on both sides, not the official CUDA kernels.
"""
import ast,hashlib,json,logging,sys
from pathlib import Path
from typing import List,Type
import torch
from torch import nn
root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'src'),str(root)]
from pointneuron.models.pointnext_native import PointNextNativeEncoder,_conv
from pointneuron.models.pointnext_ops import farthest_point_indices,ball_query_indices,gather_points,interpolate_features
torch.set_num_threads(2)
class Attributes(dict):
 __getattr__=dict.__getitem__
 __setattr__=dict.__setitem__
class Group:
 def __init__(self,cfg):self.radius=cfg['radius'];self.k=cfg['nsample']
 def __call__(self,query,support,features):
  indices=ball_query_indices(support,query,self.radius,self.k)
  dp=((gather_points(support,indices)-query[:,:,None])/self.radius).permute(0,3,1,2).contiguous()
  return dp,gather_points(features.transpose(1,2),indices).permute(0,3,1,2).contiguous()
def conv1d(a,b,norm_args=None,act_args=None,**kwargs):return _conv(a,b,1,norm=norm_args is not None,activation=act_args is not None)
def conv2d(a,b,norm_args=None,act_args=None,**kwargs):return _conv(a,b,2,norm=norm_args is not None,activation=act_args is not None)
ns={'torch':torch,'nn':nn,'List':List,'Type':Type,'logging':logging,'InvResMLP':type('InvResMLP',(),{}),
 'create_convblock1d':conv1d,'create_convblock2d':conv2d,'create_act':lambda args:nn.ReLU(),
 'CHANNEL_MAP':{'dp_fj':lambda c:c+3},'create_grouper':Group,'furthest_point_sample':farthest_point_indices,
 'get_aggregation_feautres':lambda p,dp,f,fj,feature_type:torch.cat((dp,fj),1),
 'three_interpolation':lambda q,p,f:interpolate_features(q,p,f.transpose(1,2)).transpose(1,2)}
source=root/'tmp/pointneuron2/pointnext_reference/pointnext.py'
manifest=json.loads((root/'docs/pointneuron2/pointnext_reference/reference_manifest.json').read_text())
if not source.is_file():
 raise FileNotFoundError('Save the pinned pointnext.py from reference_manifest.json to '+str(source))
if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['pointnext.py']['sha256']:
 raise ValueError('Reference source differs from the pinned hash; no comparison run')
tree=ast.parse(source.read_text())
classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ['SetAbstraction','FeaturePropogation','PointNextEncoder','PointNextDecoder']]
for node in classes:node.decorator_list=[]
exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),'pinned_openpoints_classes','exec'),ns)
reference=ns['PointNextEncoder'](in_channels=4,width=32,blocks=[1]*5,strides=[1,4,4,4,4],
 sa_layers=2,sa_use_res=True,group_args=Attributes(NAME='ballquery',normalize_dp=True),
 norm_args={'norm':'bn'},act_args={'act':'relu'},conv_args={'order':'conv-norm-act'}).eval()
decoder=ns['PointNextDecoder'](reference.channel_list).eval()
native=PointNextNativeEncoder().eval()
reference.encoder[0][0].convs[0].load_state_dict(native.stem.state_dict())
for i in range(4):
 reference.encoder[i+1][0].convs.load_state_dict(native.abstraction[i].local.state_dict())
 reference.encoder[i+1][0].skipconv.load_state_dict(native.abstraction[i].skip.state_dict())
 decoder.decoder[i][0].convs.load_state_dict(native.restoration[i].mlp.state_dict())
torch.manual_seed(10);p=torch.rand(1,358,3)*2-1;attrs=torch.cat((p,torch.rand(1,358,1)),-1)
with torch.no_grad():
 points,features=reference(p,attrs.transpose(1,2));a=decoder(points,features).transpose(1,2);b=native(p,attrs)
delta=float((a-b).abs().max());torch.testing.assert_close(a,b,rtol=0,atol=1e-6)
report={'status':'passed','scope':'pinned reference network classes with native point operations; not official CUDA kernels',
 'input_shape':[1,358,4],'output_shape':list(b.shape),'maximum_descriptor_delta':delta,
 'reference_hierarchy_counts':[x.shape[1] for x in points[1:]],
 'encoder_and_restoration_parameters':sum(p.numel() for p in native.parameters()),
 'reference_parameters':sum(p.numel() for p in reference.parameters())+sum(p.numel() for p in decoder.parameters()),
 'optimizer_updates':0,'data':'synthetic CPU coordinates and features',
 'reference_source_sha256':manifest['pointnext.py']['sha256']}
out=root/'docs/pointneuron2/pointnext_reference/topology_audit.json';out.write_text(json.dumps(report,indent=2)+'\n');print(report)
