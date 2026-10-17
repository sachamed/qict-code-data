#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
import su2_global_spin_network as gs
ROOT=Path(__file__).resolve().parents[1]
order=json.load(open(ROOT/'results'/'FRONTIER_OPTIMAL_WIDTH14_SCHEDULE.json'))['order']
pos={e:i for i,e in enumerate(order)}
# vertex path belongs to left once last incident link has crossed cut
vclose=[]
for v in gs.SITES:
 ids=[]
 for mu in range(3):
  ids += [gs.LID[(mu,*v)],gs.LID[(mu,*gs.shift(v,mu,-1))]]
 vclose.append(max(pos[e] for e in ids))

def raw_q3():
 vac=k.canonical(gs.vacuum_state());v={vac:1+0j}
 for _ in range(3):v=k.H_on_orbit_vector(v)
 raw=[]
 for rep,a in v.items():
  for st in k.orbit(rep): raw.append((st,a))
 return raw

def uid_rows(A):
 # A list of hashables -> inverse ids preserving first-seen
 d={}; ids=np.empty(len(A),np.int32)
 for i,x in enumerate(A): ids[i]=d.setdefault(x,len(d))
 return ids,d

def components(ri,ci,nr,nc):
 p=np.arange(nr+nc,dtype=np.int32);sz=np.ones(nr+nc,dtype=np.int32)
 def find(x):
  while p[x]!=x:
   p[x]=p[p[x]];x=p[x]
  return int(x)
 def union(a,b):
  a=find(a);b=find(b)
  if a==b:return
  if sz[a]<sz[b]:a,b=b,a
  p[b]=a;sz[a]+=sz[b]
 for a,b in zip(ri,ci): union(int(a),nr+int(b))
 roots=np.array([find(i) for i in range(nr+nc)],dtype=np.int32)
 _,inv=np.unique(roots,return_inverse=True)
 return inv[:nr],inv[nr:]

def profile_cut(raw,c):
 lv=[vi for vi,x in enumerate(vclose) if x<c]; rv=[vi for vi in range(8) if vi not in lv]
 R=[];C=[];D=[]
 for st,a in raw:
  R.append(tuple(st.links[e] for e in order[:c])+tuple(st.paths[vi] for vi in lv))
  C.append(tuple(st.links[e] for e in order[c:])+tuple(st.paths[vi] for vi in rv))
  D.append(a)
 ri,rmap=uid_rows(R); ci,cmap=uid_rows(C); nr=len(rmap);nc=len(cmap)
 rc,cc=components(ri,ci,nr,nc); comp_edge=rc[ri]
 sing=[]; compinfo=[]
 for comp in np.unique(comp_edge):
  mask=(comp_edge==comp); rr=ri[mask]; cl=ci[mask]; dd=np.asarray(D,complex)[mask]
  ur,rinv=np.unique(rr,return_inverse=True); uc,cinv=np.unique(cl,return_inverse=True)
  M=coo_matrix((dd,(rinv,cinv)),shape=(len(ur),len(uc))).toarray()
  s=np.linalg.svd(M,compute_uv=False)
  s=s[s>1e-13]
  sing.extend(map(float,s))
  compinfo.append((len(ur),len(uc),int(mask.sum()),len(s)))
 sing=np.array(sorted(sing,reverse=True)); fro=float(np.linalg.norm(sing)); ss=sing*sing
 tail=[]; csum=np.cumsum(ss)
 for rank in [8,16,32,64,128,256,512,1024,2048,4096,8192,16384]:
  q=min(rank,len(sing)); t=0.0 if q==len(sing) else float(np.sqrt(max(0.0,fro*fro-(float(csum[q-1]) if q else 0.0))))
  tail.append({'rank':rank,'discarded_l2':t,'fraction':t/fro if fro else 0.0})
 # ranks required for absolute targets (normalized state scale: report raw and normalized)
 req={}
 for eps in [1e-2,1e-4,1e-6,1e-8,1e-10]:
  threshold=(eps*fro)**2
  total=fro*fro
  # smallest q with tail^2 <= threshold
  q=int(np.searchsorted(csum,total-threshold,side='left')+1)
  q=min(q,len(sing))
  req[str(eps)]={'rank':q,'normalized_discarded_l2':0.0 if q==len(sing) else (float(np.sqrt(max(0,total-(csum[q-1] if q else 0)))/fro) if fro else 0)}
 return {
  'cut':c,'rows':nr,'cols':nc,'nnz':len(D),'components':len(compinfo),'exact_schmidt_rank':len(sing),
  'largest_component_rows':max(x[0] for x in compinfo),'largest_component_cols':max(x[1] for x in compinfo),
  'largest_component_nnz':max(x[2] for x in compinfo),'largest_component_rank':max(x[3] for x in compinfo),
  'fro_norm':fro,'top32_singular_values':sing[:32].tolist(),'tails':tail,'required_rank_fractional':req}

def main():
 t=time.time();raw=raw_q3();rows=[]
 for c in range(1,24):
  r=profile_cut(raw,c);rows.append(r)
  print(c,r['components'],r['exact_schmidt_rank'],r['largest_component_rows'],r['largest_component_cols'],r['required_rank_fractional']['1e-06']['rank'],flush=True)
 out={'status':'PASS_SU2_Q3_EXACT_COMPONENT_SCHMIDT_PROFILE','raw_nonzeros':len(raw),'order':order,'rows':rows,'seconds':time.time()-t,
      'scope':'Exact component-wise Schmidt spectrum of H_B^3|vac> in the fusion-tree spin-network basis along the certified width-14 link order. No physical h_d is inferred.'}
 (ROOT/'results'/'SU2_Q3_COMPONENT_SCHMIDT_PROFILE.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'seconds':out['seconds'],'worst_exact_rank':max(x['exact_schmidt_rank'] for x in rows),'worst_rank_1e-6':max(x['required_rank_fractional']['1e-06']['rank'] for x in rows),'max_component_rows':max(x['largest_component_rows'] for x in rows),'max_component_cols':max(x['largest_component_cols'] for x in rows)},indent=2))
if __name__=='__main__':main()
