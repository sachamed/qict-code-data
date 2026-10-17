#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,time,collections,math
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import eigsh
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
import su2_global_spin_network as gs
ROOT=Path(__file__).resolve().parents[1]

def ecost(st): return sum(n*(n+2) for n in st.links)

def local_neighbors(rep):
 out=collections.defaultdict(complex)
 for pid in range(24):
  for tgt,z in k.wp_action(rep,pid): out[k.canonical(tgt)] += z
 return {s:z for s,z in out.items() if abs(z)>1e-14}

def basis(Emax,max_states=200000):
 vac=k.canonical(gs.vacuum_state()); seen={vac};q=collections.deque([vac])
 while q:
  s=q.popleft()
  for t in local_neighbors(s):
   if ecost(t)<=Emax and t not in seen:
    seen.add(t);q.append(t)
    if len(seen)>max_states:raise RuntimeError('basis too large')
 return sorted(seen,key=lambda s:(ecost(s),k.skey(s)))

def run(Emax):
 t0=time.time();B=basis(Emax);N=len(B);idx={s:i for i,s in enumerate(B)};sizes=np.array([len(k.orbit(s)) for s in B],float)
 rr=[];cc=[];dd=[];outside_edges=0
 neigh_cache=[]
 for a,s in enumerate(B):
  na=sizes[a]; loc=local_neighbors(s);neigh_cache.append(loc)
  for t,z in loc.items():
   b=idx.get(t)
   if b is None:outside_edges+=1;continue
   rr.append(b);cc.append(a);dd.append(math.sqrt(na/sizes[b])*z)
 H=coo_matrix((np.asarray(dd,complex),(rr,cc)),shape=(N,N)).tocsr();H.sum_duplicates()
 herm=float(np.linalg.norm((H-H.getH()).data)) if (H-H.getH()).nnz else 0.0
 KE=np.array([ecost(s) for s in B],float);K=diags(KE,format='csr')-16.0*H
 nev=min(4,max(1,N-1));w,V=eigsh(K,k=nev,which='SA',tol=1e-11,maxiter=20000);o=np.argsort(w);w=w[o];V=V[:,o]
 rows=[]
 for j in range(min(3,len(w))):
  c=V[:,j]; inside=float(np.linalg.norm(K@c-w[j]*c)); out=collections.defaultdict(complex)
  for a,s in enumerate(B):
   if abs(c[a])<1e-15:continue
   na=sizes[a]
   for t,z in neigh_cache[a].items():
    if t in idx:continue
    nb=len(k.orbit(t)); out[t] += -16.0*math.sqrt(na/nb)*z*c[a]
  outside=float(math.sqrt(sum(abs(z)**2 for z in out.values())))
  total=math.hypot(inside,outside)
  rows.append({'ritz_value_no_constant':float(w[j]),'inside_residual':inside,'outside_residual':outside,'full_residual_bound':total,'outside_orbits':len(out),'vacuum_overlap_abs':float(abs(c[0]))})
 gap=float(w[1]-w[0]) if len(w)>1 else None
 out={'status':'PASS_SU2_ENERGY_CUT_RITZ_RESIDUAL','Emax':Emax,'orbit_basis_dim':N,'H_nnz':int(H.nnz),'outside_directed_edges':outside_edges,'hermiticity_data_norm':herm,'lowest_ritz':rows,'ritz_gap':gap,'seconds':time.time()-t0,
      'scope':'Exact translation-orbit Galerkin Ritz test for Kprim_SU2=sum n(n+2)-16H_B. The outside residual is computed explicitly from all fundamental-plaquette transitions leaving the electric-cost subspace. This diagnoses a possible Kato-state Galerkin route; it is not the Floquet h_d contraction.'}
 (ROOT/'results'/f'SU2_ENERGY_CUT_RITZ_E{Emax}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run(int(sys.argv[1]))
