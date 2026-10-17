#!/usr/bin/env python3
"""Exact sparse modular Poisson test for global static U1 charge reduction.

No dense cycle Gram matrix is formed. Field obstructions are retained as
verifiable null vectors; successful lifts are checked on every actual link.
"""
from pathlib import Path
from collections import deque
import hashlib,json,heapq,time,gc
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/global_modular_charge'
PAYLOAD=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
BASIS=ROOT/'reproducibility/results/c10_primitive_gauss/C10_INTEGER_GAUSS_BASIS_AND_SOURCE_PATHS.npz'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class ModularFactor:
    def __init__(self,matrix,p):
        self.p=p;self.n=matrix.shape[0];self.history=[];self.free=[]
        A=matrix.tocsr();rows=[]
        for i in range(self.n):
            row={int(j):int(v)%p for j,v in zip(A.indices[A.indptr[i]:A.indptr[i+1]],
                    A.data[A.indptr[i]:A.indptr[i+1]]) if int(v)%p}
            rows.append(row)
        active=set(range(self.n));heap=[(len(row),i) for i,row in enumerate(rows)]
        heapq.heapify(heap);start=time.monotonic();processed=0;peak=0
        while active:
            while True:
                degree,v=heapq.heappop(heap)
                if v in active and degree==len(rows[v]):break
            if not rows[v]:
                self.free.append(v);active.remove(v);processed+=1;continue
            if rows[v].get(v,0):
                pivots=(v,);inv=((pow(rows[v][v],-1,p),),)
            else:
                w=min((j for j in rows[v] if j!=v),key=lambda j:len(rows[j]))
                a=rows[v][w];b=rows[w].get(w,0);ai=pow(a,-1,p)
                pivots=(v,w);inv=(((-b*ai*ai)%p,ai),(ai,0))
            neighbors=sorted(set().union(*(rows[j].keys() for j in pivots))-set(pivots))
            vectors=[tuple(rows[u].get(j,0) for j in pivots) for u in neighbors]
            self.history.append((pivots,inv,tuple(zip(neighbors,vectors))))
            k=len(pivots);transformed=[]
            for b in vectors:
                transformed.append(tuple(sum(b[i]*inv[i][j] for i in range(k))%p
                                         for j in range(k)))
            for u in neighbors:
                for j in pivots:rows[u].pop(j,None)
            for i,u in enumerate(neighbors):
                cu=transformed[i]
                for j in range(i,len(neighbors)):
                    w=neighbors[j];bw=vectors[j]
                    delta=sum(cu[a]*bw[a] for a in range(k))
                    new=(rows[u].get(w,0)-delta)%p
                    if new:rows[u][w]=new;rows[w][u]=new
                    else:rows[u].pop(w,None);rows[w].pop(u,None)
            for j in pivots:rows[j].clear();active.remove(j)
            for u in neighbors:heapq.heappush(heap,(len(rows[u]),u))
            processed+=k;peak=max(peak,len(neighbors))
            if processed//500!=(processed-k)//500:
                print(json.dumps({'field':p,'eliminated':processed,
                  'active':len(active),'largest_front':peak,
                  'seconds':round(time.monotonic()-start,2)}),flush=True)
        self.rank=self.n-len(self.free);self.peak=peak

    def apply(self,rhs,free_values=None):
        p=self.p;b=np.asarray(rhs,dtype=np.int64).copy()
        if b.ndim==1:b=b[:,None]
        b%=p;y=np.zeros_like(b)
        for pivots,inv,external in self.history:
            ids=list(pivots);z=(np.asarray(inv,dtype=np.int64)@b[ids])%p
            y[ids]=z
            for u,coef in external:b[u]=(b[u]-np.asarray(coef)@z)%p
        constraints=b[self.free].copy()
        x=np.zeros_like(b)
        if free_values is not None:x[self.free]=free_values
        for pivots,inv,external in reversed(self.history):
            k=len(pivots);total=np.zeros((k,b.shape[1]),np.int64)
            for u,coef in external:total+=np.asarray(coef)[:,None]*x[u]
            x[list(pivots)]=(y[list(pivots)]-np.asarray(inv)@total)%p
        return x,constraints

    def kernel(self):
        if not self.free:return np.zeros((self.n,0),np.int64)
        return self.apply(np.zeros((self.n,len(self.free)),np.int64),
                          np.eye(len(self.free),dtype=np.int64))[0]


def small_solve(A,b,p):
    A=np.asarray(A,dtype=np.int64)%p;b=np.asarray(b,dtype=np.int64)%p
    if b.ndim==1:b=b[:,None]
    aug=np.column_stack([A,b]);r=0;piv=[];n=A.shape[1]
    for col in range(n):
        candidates=np.flatnonzero(aug[r:,col])
        if not len(candidates):continue
        j=r+int(candidates[0]);aug[[r,j]]=aug[[j,r]]
        aug[r]=(aug[r]*pow(int(aug[r,col]),-1,p))%p
        for i in range(len(aug)):
            if i!=r and aug[i,col]:aug[i]=(aug[i]-aug[i,col]*aug[r])%p
        piv.append(col);r+=1
        if r==len(aug):break
    bad=np.any(aug[r:,n:]!=0,axis=0)
    x=np.zeros((n,b.shape[1]),np.int64)
    for i,col in enumerate(piv):x[col]=aug[i,n:]
    return x,bad


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(PAYLOAD,allow_pickle=False) as z:links=z['links'].astype(int)
    with np.load(BASIS,allow_pickle=False) as z:
        vertices=z['gauge_vertices'].astype(int);tree=z['spanning_tree_links'].astype(int)
        chords=z['chord_links'].astype(int);paths={r:z[f'path_R{r}'].astype(int) for r in [3,5,7,9]}
    index={tuple(v):i for i,v in enumerate(vertices)};starts=links[:,1:];ends=starts.copy()
    ends[np.arange(len(links)),links[:,0]]+=1
    tail=np.array([index[tuple(x)] for x in starts]);head=np.array([index[tuple(x)] for x in ends])
    row=np.r_[tail,head];col=np.tile(np.arange(len(links)),2)
    incidence=sparse.csr_matrix((np.r_[-np.ones(len(links),int),np.ones(len(links),int)],
                                 (row,col)),shape=(len(vertices),len(links)))
    root=index[(0,0,0)];keep=np.array([i for i in range(len(vertices)) if i!=root])
    lap=(incidence@incidence.T).tocsr();A=lap[keep][:,keep].tocsr()
    H=np.zeros((len(links),len(paths)),np.int64)
    for j,path in enumerate(paths.values()):H[path,j]=1
    bfull=incidence@H;rhs=(3*bfull)[keep]
    records=[];arrays={'gauge_vertices':vertices,'source_radii':np.array(list(paths)),
                       'source_path_vectors':H.astype(np.int8)}
    factor7=ModularFactor(A,7);sol7,con7=factor7.apply(rhs)
    K7=factor7.kernel();assert not np.any((A@K7)%7)
    eligible=[]
    for j,radius in enumerate(paths):
        valid=not np.any(con7[:,j])
        rec={'radius':radius,'mod7_rank':factor7.rank,'mod7_nullity':K7.shape[1],
             'mod7_largest_front':factor7.peak,'mod7_source_solvable':valid}
        if valid:
            assert not np.any((A@sol7[:,j]-rhs[:,j])%7);eligible.append(j)
        else:
            bad=int(np.flatnonzero(con7[:,j])[0]);witness=K7[:,bad]
            pairing=int(witness@rhs[:,j]%7);assert pairing!=0
            full=np.zeros(len(vertices),np.int64);full[keep]=witness
            assert not np.any((lap@full)%7)
            arrays[f'R{radius}_obstruction_mod7']=full.astype(np.int8)
            rec['exact_obstruction_pairing_mod7']=pairing
            rec['global_flux_translation_ansatz_exists']=False
        records.append(rec)
    del factor7,K7;gc.collect()
    if eligible:
        factor3=ModularFactor(A,3);K=factor3.kernel();assert not np.any((A@K)%3)
        AK=A@K;assert not np.any(AK%3)
        B=(AK//3)%3;M=(K.T@B)%3
        target=bfull[keep][:,eligible]%3
        weights,bad=small_solve(M,(K.T@target)%3,3)
        adjusted=(target-B@weights)%3
        psi,constraints=factor3.apply(adjusted)
        for offset,j in enumerate(eligible):
            rec=records[j];rec['mod3_rank']=factor3.rank;rec['mod3_nullity']=K.shape[1]
            rec['mod3_largest_front']=factor3.peak
            if bad[offset] or np.any(constraints[:,offset]):
                rec['mod9_source_solvable']=False
                rec['global_flux_translation_ansatz_exists']=False
                rec['mod9_obstruction_scope']='Exact Bockstein compatibility failed; no fitted phase or potential supplied.'
                continue
            x9=(K@weights[:,offset]+3*psi[:,offset])%9
            assert not np.any((A@x9-rhs[:,j])%9)
            x7=sol7[:,j]
            potential=(x7+7*((4*(x9-x7))%9))%63
            full=np.zeros(len(vertices),np.int64);full[keep]=potential
            assert not np.any((lap@full-3*bfull[:,j])%63)
            f=(-3*H[:,j]+incidence.T@full)%63
            assert not np.any((incidence@f)%63)
            # Complete chord fluxes on the fixed integer spanning tree.
            graph=[[] for _ in vertices]
            for edge in tree:
                u,v=int(tail[edge]),int(head[edge])
                graph[u].append((v,int(edge)));graph[v].append((u,int(edge)))
            parent=np.full(len(vertices),-1,int);parent[root]=root;edge_parent=np.full(len(vertices),-1,int)
            order=[root]
            for u in order:
                for v,e in graph[u]:
                    if parent[v]<0:parent[v]=u;edge_parent[v]=e;order.append(v)
            assert len(order)==len(vertices)
            flow=np.zeros(len(links),np.int64);flow[chords]=f[chords]
            divergence=incidence@flow
            for v in reversed(order[1:]):
                e=edge_parent[v];sign=1 if head[e]==v else -1
                flow[e]=-sign*divergence[v];divergence[parent[v]]+=divergence[v];divergence[v]=0
            assert not np.any(incidence@flow) and int(flow.sum())==0
            assert not np.any((flow+3*H[:,j]-incidence.T@full)%63)
            radius=rec['radius']
            constant=(2*sum(int(x)**2 for x in flow)-15*radius)%126
            arrays[f'R{radius}_potential_mod63']=full.astype(np.int16)
            arrays[f'R{radius}_neutral_integer_cycle_flux']=flow
            arrays[f'R{radius}_chord_translation']=flow[chords]
            rec.update({'mod9_source_solvable':True,'global_flux_translation_ansatz_exists':True,
               'mod63_Poisson_residual':0,'integer_Gauss_residual':0,
               'charge_qY':-3,'coherent_charge_phase_mod126':int(constant),
               'right_module_charge_qY':-6,
               'right_module_phase_mod126':int((8*sum(int(x)**2 for x in flow)-66*radius)%126),
               'max_integer_flux_translation':int(np.max(abs(flow)))})
        del factor3;gc.collect()
    npz=OUT/'GLOBAL_MODULAR_CHARGE_SOLUTIONS_AND_OBSTRUCTIONS.npz'
    np.savez_compressed(npz,**arrays)
    result={'status':'EXACT_GLOBAL_MODULAR_POISSON_CHARGE_REDUCTION_TEST',
      'vertices':len(vertices),'links':len(links),'cycle_rank':len(chords),
      'charges':[-3,-6],'modulus':63,'CRT_moduli':[7,9],
      'source_root':vertices[root].tolist(),'source_paths':records,
      'criterion':'G*delta=q*B^T*h mod 63 iff the rooted graph Laplacian solves Lap*phi=-q*partial(h) mod 63.',
      'mod9_method':'GF3 kernel and exact Bockstein matrix K^T*(A*K/3), followed by lifting and CRT.',
      'source_boundary_scope':'The four authentic dressing paths are tested on the fixed whole gauge support; they are not four spatial Kato buffers.',
      'source_evolution_scope':'Static U1 electric/plaquette transfer only. Any moving-source stream must retain the transformed Wilson and CAR sources coherently.',
      'no_dense_cycle_Gram_matrix':True,'all_arithmetic_exact':True,
      'payload_sha256':sha(PAYLOAD),'integer_basis_sha256':sha(BASIS),
      'full_C10_contract_completed':False,'data_sha256':sha(npz)}
    (OUT/'GLOBAL_MODULAR_CHARGE_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'source_paths':records,'full_C10':False},indent=2))


if __name__=='__main__':main()
