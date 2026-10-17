#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
import su2_global_spin_network as gs
ROOT=Path(__file__).resolve().parents[1]
order=json.load(open(ROOT/'results'/'FRONTIER_OPTIMAL_WIDTH14_SCHEDULE.json'))['order']
pos={e:i for i,e in enumerate(order)}
# vertex incident links and closure position in this ordering
vlinks=[];vclose=[]
for v in gs.SITES:
 ids=[]
 for mu in range(3):
  ids.append(gs.LID[(mu,*v)])
  sm=gs.shift(v,mu,-1);ids.append(gs.LID[(mu,*sm)])
 vlinks.append(tuple(ids));vclose.append(max(pos[e] for e in ids))

def raw_q3():
 vac=k.canonical(gs.vacuum_state());v={vac:1+0j}
 for _ in range(3):v=k.H_on_orbit_vector(v)
 raw=[]
 for rep,a in v.items():
  for st in k.orbit(rep):raw.append((st,a))
 return raw

def matrix_for_cut(raw,c):
 # c = number processed links, 1..23. Vertex fusion-path categorical variable is assigned left iff all 6 incident links processed.
 Lset=set(order[:c]); Rset=set(order[c:])
 left_vertices=[vi for vi,x in enumerate(vclose) if x < c]
 right_vertices=[vi for vi in range(8) if vi not in left_vertices]
 rmap={};cmap={}; rr=[];cc=[];dd=[]
 for st,a in raw:
  rk=tuple(st.links[e] for e in order[:c])+tuple(st.paths[vi] for vi in left_vertices)
  ck=tuple(st.links[e] for e in order[c:])+tuple(st.paths[vi] for vi in right_vertices)
  i=rmap.setdefault(rk,len(rmap)); j=cmap.setdefault(ck,len(cmap));rr.append(i);cc.append(j);dd.append(a)
 M=coo_matrix((np.asarray(dd,complex),(rr,cc)),shape=(len(rmap),len(cmap))).tocsr();M.sum_duplicates()
 return M,left_vertices

def top_profile(M,Ks=(16,32,64,128)):
 fro2=float(np.sum(np.abs(M.data)**2)); mn=min(M.shape)
 # ARPACK requires k < min(M.shape)-1.  For small cuts the exact dense
 # spectrum is cheaper and avoids a spurious k>=N-1 failure.
 if mn<=1:
  s=np.array([np.sqrt(fro2)]) if fro2 else np.array([])
 elif mn<=max(Ks)+2 and max(M.shape)<=12000:
  s=np.linalg.svd(M.toarray(),compute_uv=False)
 else:
  kmax=min(max(Ks),mn-2)
  try:
   s=svds(M,k=kmax,return_singular_vectors=False,which='LM',tol=1e-10,maxiter=20000)
   s=np.sort(np.abs(s))[::-1]
  except Exception:
   if max(M.shape)<=3000:
    s=np.linalg.svd(M.toarray(),compute_uv=False)
   else: raise
 out={}
 for K in Ks:
  q=min(K,len(s)); kept=float(np.sum(s[:q]**2)); tail=max(0.0,fro2-kept)**0.5
  out[str(K)]={'discarded_norm':tail,'discarded_fraction_of_fro_norm':tail/(fro2**0.5) if fro2 else 0.0,'captured_probability_fraction':kept/fro2 if fro2 else 1.0}
 return fro2**0.5,out, [float(x) for x in s[:min(16,len(s))]]

def main():
 t=time.time();raw=raw_q3(); rows=[]
 total=np.sqrt(sum(abs(a)**2 for _,a in raw))
 for c in range(1,24):
  M,lv=matrix_for_cut(raw,c); fn,p,top=top_profile(M)
  rows.append({'cut':c,'processed_link':order[c-1],'matrix_shape':list(M.shape),'nnz':int(M.nnz),'closed_vertices':lv,'fro_norm':fn,'profiles':p,'top16_singular_values':top})
  print(c,M.shape,M.nnz,p['64']['discarded_fraction_of_fro_norm'],flush=True)
 out={'status':'PASS_SU2_Q3_FRONTIER_SCHMIDT_PROFILE','state':'H_B^3|vac> exact translation-symmetric spin-network vector','raw_spin_network_nonzeros':len(raw),'norm':float(total),'order':order,'rows':rows,'seconds':time.time()-t,
      'scope':'Diagnostic of SVD compressibility in an exact categorical spin-network coefficient tensor, assigning each vertex fusion-path label to the side containing its last incident link. It does not yet apply the PEPO gate or evaluate h_d.'}
 (ROOT/'results'/'SU2_Q3_FRONTIER_SCHMIDT_PROFILE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'raw':len(raw),'seconds':out['seconds'],'worst_tail64':max(r['profiles']['64']['discarded_fraction_of_fro_norm'] for r in rows),'worst_tail128':max(r['profiles']['128']['discarded_fraction_of_fro_norm'] for r in rows)},indent=2))
if __name__=='__main__':main()
