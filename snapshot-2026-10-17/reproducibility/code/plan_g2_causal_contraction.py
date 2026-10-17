#!/usr/bin/env python3
"""L24 C10 support-graph sweep planner; geometry, NOT quantum bond dimension."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
ROOT=Path(__file__).resolve().parents[1]
PAY=ROOT/'data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
OUT=ROOT/'results/G2_C10_SWEEP_PLANNER.json'

def scan(A,order):
    n=len(order); inverse=np.empty(n,int);inverse[order]=np.arange(n)
    farthest=np.arange(n,dtype=int)
    for site in range(n):
        neigh=A.indices[A.indptr[site]:A.indptr[site+1]]
        farthest[inverse[site]]=max([int(inverse[site]),*inverse[neigh].tolist()])
    change=np.zeros(n+1,int)
    for i,f in enumerate(farthest):
        if f>i:change[i]+=1;change[f]-=1
    front=np.cumsum(change)[:-1]
    return {'maximum_vertex_frontier':int(front.max()),
            'median_vertex_frontier':float(np.median(front)),
            'max_edge_bandwidth':int(max((abs(inverse[A.nonzero()[0]]-inverse[A.nonzero()[1]])),default=0)),
            'frontier_at_halfway':int(front[n//2])}

def main():
    with np.load(PAY,allow_pickle=False) as z:
        n=len(z['c10_nodes']);src=z['c10_edge_src'].astype(int);dst=z['c10_edge_dst'].astype(int)
        mask=(src>=0)&(dst>=0)&(src<n)&(dst<n)
        a=src[mask];b=dst[mask]
        A=coo_matrix((np.ones(2*len(a),dtype=np.int8),
                      (np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr(); A.data[:]=1;A.setdiag(0);A.eliminate_zeros()
        order_rcm=reverse_cuthill_mckee(A,symmetric_mode=True)
        results={'as_serialized':scan(A,np.arange(n)), 'reverse_cuthill_mckee':scan(A,order_rcm)}
        best='as_serialized' if results['as_serialized']['maximum_vertex_frontier']<=results['reverse_cuthill_mckee']['maximum_vertex_frontier'] else 'reverse_cuthill_mckee'
        out={'classification':'EXACT_C10_COMBINATORIAL_SUPPORT_GRAPH_ANALYSIS_ONLY',
             'nodes':int(n),'directed_edges':int(len(src)),'internal_directed_edges':int(mask.sum()),
             'boundary_directed_edges_with_negative_endpoint':int((~mask).sum()),
             'candidate_orderings':results,'min_frontier_of_two_tested_orderings':best,
             'no_claim_of_optimal_ordering':True,'physical_g2_promoted':False,
             'scope':'Vertex cutwidth heuristic on serialized C10 population graph. Quantum bond dimensions, iota path contractions and coherent amplitudes are separate input data.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'PASS_GEOMETRY_SWEEP','n_vertices':n,'candidate_max_frontiers':{k:v['maximum_vertex_frontier'] for k,v in results.items()},'best_tested':best},indent=2))

if __name__=='__main__':main()
