"""Training-only trace-agreement labels. Never called by candidate construction."""
import numpy as np
from scipy.spatial import cKDTree

from pn2.evaluation.reference_shortcuts import ReferencePaths
from pn2.supervision.curve_nodes import _chain_geometry

NODE_REASONS={'trace_unique':0,'farther_than_2':1,'projection_ambiguous':2,'reference_empty':3}
EDGE_REASONS={'consecutive':0,'touches_negative':1,'unknown_node':2,'fragment_relation_unknown':3,
              'isolated_reference_unknown':4,'coincident_projections_unknown':5,'over_32':6,
              'nonconsecutive':7,'junction_transition_unknown':8,'reference_empty':9}


class TrainingPaths(ReferencePaths):
    def __init__(self,morphology):
        super().__init__(morphology)
        self.m=morphology
        self.directions=[[] for _ in self.xyz]
        for u,v in self.edges:
            self.directions[u].append(self.xyz[v]-self.xyz[u])
            self.directions[v].append(self.xyz[u]-self.xyz[v])
        self.isolated=np.flatnonzero(morphology.degrees==0)
        self.isolated_tree=cKDTree(self.xyz[self.isolated]) if len(self.isolated) else None
        _,_,chains=_chain_geometry(morphology)
        self.chain_info=chains
        self.edge_chain={};self.vertex_chains={}
        by_pair={tuple(sorted(map(int,e))):i for i,e in enumerate(self.edges)}
        for ci,(ids,points,radii,arc) in enumerate(chains):
            self.vertex_chains.setdefault(ids[0],[]).append((ci,0.))
            self.vertex_chains.setdefault(ids[-1],[]).append((ci,float(arc[-1])))
            for a,b,aa,bb in zip(ids[:-1],ids[1:],arc[:-1],arc[1:]):
                ei=by_pair[tuple(sorted((a,b)))];u,v=self.edges[ei]
                self.edge_chain[ei]=(ci,float(aa if u==a else bb),float(bb if v==b else aa))

    def project(self,point):
        candidates=[]
        if self.index is not None:
            idx=self.index
            raw=idx.tree.query_ball_point(point,2.25+idx.half+1e-9)
            ids=np.unique(idx.parents[raw]) if len(raw) else np.empty(0,int)
            if len(ids):
                starts=idx.segments[ids,0];delta=idx.segments[ids,1]-starts
                fraction=np.clip(np.einsum('ij,ij->i',point-starts,delta)/np.maximum((delta*delta).sum(1),1e-30),0,1)
                projected=starts+fraction[:,None]*delta
                distance=np.linalg.norm(point-projected,axis=1)
                for j,ei in enumerate(ids):
                    t=float(fraction[j]);q=projected[j]
                    if t in (0.,1.):
                        vertex=int(self.edges[ei,int(t)])
                        directions=np.asarray(self.directions[vertex])
                        # Topological incident directions only: never borrow a
                        # direction from a coincident disconnected component.
                        if len(directions) and np.any(directions@(point-self.xyz[vertex])>1e-10):continue
                    candidates.append((float(distance[j]),'edge',int(ei),t,q))
        if self.isolated_tree is not None:
            for i in self.isolated_tree.query_ball_point(point,2.25+1e-9):
                vertex=int(self.isolated[i]);q=self.xyz[vertex]
                candidates.append((float(np.linalg.norm(point-q)),'isolated',vertex,0.,q))
        if not candidates:return 0,None,'farther_than_2'
        candidates.sort(key=lambda x:(x[0],x[1],x[2],x[3]));best=candidates[0]
        if best[0]>2.:return 0,None,'farther_than_2'
        def separation(a,b):
            if a[1]=='edge' and b[1]=='edge':return self.path((a[2],a[3],a[4]),(b[2],b[3],b[4]))[0]
            return 0. if a[1]==b[1]=='isolated' and a[2]==b[2] else np.inf
        if any(separation(best,other)>.5+1e-12 for other in candidates if other[0]<=best[0]+.25+1e-12):
            return -1,None,'projection_ambiguous'
        return 1,best,'trace_unique'

    def projection_vertex_distance(self,projection,vertex):
        if projection[1]=='isolated':return self.route(projection[2],vertex)
        ei,t=projection[2],projection[3];u,v=self.edges[ei];length=self.lengths[ei]
        return min(t*length+self.route(u,vertex),(1-t)*length+self.route(v,vertex))


def label_candidates(points,edges,morphology,*,partition):
    if partition!='training':raise ValueError('Candidate labels are training-only; development must never enter this API')
    p=np.asarray(points,float).reshape(-1,3);edges=np.asarray(edges,int).reshape(-1,2)
    if not np.isfinite(p).all() or (len(edges) and (edges.min()<0 or edges.max()>=len(p))):
        raise ValueError('Invalid candidate geometry')
    index=TrainingPaths(morphology)
    if not len(morphology.xyz):
        return {'node_label':np.full(len(p),-1,np.int8),'edge_label':np.full(len(edges),-1,np.int8),
                'node_reason':np.full(len(p),NODE_REASONS['reference_empty'],np.int8),
                'edge_reason':np.full(len(edges),EDGE_REASONS['reference_empty'],np.int8),
                'junction_anchor_vertex':np.full(len(p),-1,int)}, {'junctions':0,'junctions_with_anchor':0}
    projected=[index.project(point) for point in p]
    node=np.array([r[0] for r in projected],np.int8)
    node_reason=np.array([NODE_REASONS[r[2]] for r in projected],np.int8)
    memberships={};components=np.full(len(p),-1,int)
    for i,(label,q,reason) in enumerate(projected):
        if label!=1:continue
        if q[1]=='isolated':
            components[i]=index.component[q[2]];memberships[i]=[];continue
        ei,t=q[2],q[3];u,v=index.edges[ei];components[i]=index.component[u]
        ci,a,b=index.edge_chain[ei]
        vertex=int(index.edges[ei,int(t)]) if t in (0.,1.) else None
        memberships[i]=list(index.vertex_chains[vertex]) if vertex in index.vertex_chains else [(ci,a+t*(b-a))]
    # Assign one distinct positive candidate to each degree>=3 junction, at
    # <=2 Euclidean AND reference-path distance. This labels a representation;
    # it never creates/moves/selects inference candidates or edges.
    anchors=np.full(len(p),-1,int);positive=np.flatnonzero(node==1)
    tree=cKDTree(p[positive]) if len(positive) else None
    junctions=np.flatnonzero(morphology.degrees>=3)
    for vertex in sorted(junctions,key=lambda v:int(morphology.ids[v])):
        candidates=[] if tree is None else positive[tree.query_ball_point(index.xyz[vertex],2.)]
        choices=[]
        for i in candidates:
            if anchors[i]>=0 or components[i]!=index.component[vertex]:continue
            if index.projection_vertex_distance(projected[i][1],int(vertex))<=2.+1e-12:
                choices.append((float(np.linalg.norm(p[i]-index.xyz[vertex])),int(i)))
        if choices:
            _,i=min(choices);anchors[i]=vertex;memberships[i]=list(index.vertex_chains.get(int(vertex),[]))
    chain_nodes={}
    for i,items in memberships.items():
        for ci,arc in items:chain_nodes.setdefault(ci,[]).append((arc,i))
    consecutive=set();coincident=set()
    for ci,items in chain_nodes.items():
        ordered=sorted(set(items));groups=[]
        for arc,i in ordered:
            if not groups or arc-groups[-1][0][0]>.5:groups.append([])
            groups[-1].append((arc,i))
        for group in groups:
            ids=sorted({i for _,i in group})
            for a,i in enumerate(ids):
                for j in ids[a+1:]:coincident.add((i,j))
        for left,right in zip(groups[:-1],groups[1:]):
            for aa,i in left:
                for bb,j in right:
                    if i!=j and 0<bb-aa<=32.:consecutive.add(tuple(sorted((i,j))))
    # Bulk assignment for the common, clearly negative/masked cases.
    labels=np.full(len(edges),-1,np.int8);reasons=np.full(len(edges),EDGE_REASONS['unknown_node'],np.int8)
    touched_negative=np.any(node[edges]==0,axis=1)
    labels[touched_negative]=0;reasons[touched_negative]=EDGE_REASONS['touches_negative']
    known=np.flatnonzero(np.all(node[edges]==1,axis=1))
    for ei in known:
        u,v=map(int,edges[ei]);pair=tuple(sorted((u,v)))
        if components[u]!=components[v]:reasons[ei]=EDGE_REASONS['fragment_relation_unknown'];continue
        a,b=dict(memberships[u]),dict(memberships[v]);shared=set(a)&set(b)
        if not a or not b:reasons[ei]=EDGE_REASONS['isolated_reference_unknown'];continue
        if pair in coincident:reasons[ei]=EDGE_REASONS['coincident_projections_unknown'];continue
        if shared:
            length=min(abs(a[ci]-b[ci]) for ci in shared)
            if length>32.:labels[ei]=0;reasons[ei]=EDGE_REASONS['over_32']
            elif pair in consecutive:labels[ei]=1;reasons[ei]=EDGE_REASONS['consecutive']
            else:labels[ei]=0;reasons[ei]=EDGE_REASONS['nonconsecutive']
        else:
            endpoints_a={int(index.chain_info[ci][0][j]) for ci in a for j in (0,-1)}
            endpoints_b={int(index.chain_info[ci][0][j]) for ci in b for j in (0,-1)}
            if endpoints_a&endpoints_b:reasons[ei]=EDGE_REASONS['junction_transition_unknown']
            else:labels[ei]=0;reasons[ei]=EDGE_REASONS['nonconsecutive']
    result={'node_label':node,'edge_label':labels,'node_reason':node_reason,'edge_reason':reasons,
            'junction_anchor_vertex':anchors}
    stats={'junctions':int(len(junctions)),'junctions_with_anchor':int(np.count_nonzero(anchors>=0)),
           'reference_components':int(morphology.component_count),'path_bound_voxel':32.,'positions_moved':False}
    return result,stats


def label_counts(arrays):
    result={}
    for kind,codes in [('node',NODE_REASONS),('edge',EDGE_REASONS)]:
        values=arrays[kind+'_label'];n=len(values)
        result[kind]={'total':n,'positive':int((values==1).sum()),'negative':int((values==0).sum()),
                      'unknown':int((values<0).sum()),'masked_fraction':float((values<0).mean()) if n else None,
                      'reasons':{name:int((arrays[kind+'_reason']==code).sum()) for name,code in codes.items()}}
    return result
