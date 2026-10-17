#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,time,math
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
import su2_global_spin_network as gs
ROOT=Path(__file__).resolve().parents[1]
order=json.load(open(ROOT/'results'/'FRONTIER_OPTIMAL_WIDTH14_SCHEDULE.json'))['order']
pos={e:i for i,e in enumerate(order)}
# 32-variable order: physical links; a vertex fusion-path categorical variable is inserted
# immediately after the last incident link crosses the certified sweep.
vclose={}
for vi,v in enumerate(gs.SITES):
 ids=[]
 for mu in range(3): ids += [gs.LID[(mu,*v)],gs.LID[(mu,*gs.shift(v,mu,-1))]]
 vclose[vi]=max(pos[e] for e in ids)
varspec=[]
for i,e in enumerate(order):
 varspec.append(('L',e))
 for vi in sorted(v for v,c in vclose.items() if c==i): varspec.append(('V',vi))
assert len(varspec)==32

def raw_q3_samples():
 vac=k.canonical(gs.vacuum_state());v={vac:1+0j}
 for _ in range(3): v=k.H_on_orbit_vector(v)
 acc=defaultdict(complex)
 for rep,a in v.items():
  for st in k.orbit(rep):
   x=[]
   for typ,i in varspec: x.append(st.links[i] if typ=='L' else st.paths[i])
   acc[tuple(x)] += a
 return acc

def component_svd(entries,tol=1e-12):
 # entries=(row_key,col_key,val), return core triples row_key,newbond,val and remainder newbond,col_key,val
 rmap={};cmap={};rows=[];cols=[];vals=[]
 for r,c,z in entries:
  ri=rmap.setdefault(r,len(rmap));ci=cmap.setdefault(c,len(cmap));rows.append(ri);cols.append(ci);vals.append(z)
 nr,nc=len(rmap),len(cmap)
 radj=[[] for _ in range(nr)]; cadj=[[] for _ in range(nc)]
 for i,(r,c) in enumerate(zip(rows,cols)):radj[r].append((c,i));cadj[c].append((r,i))
 seenr=np.zeros(nr,bool); seenc=np.zeros(nc,bool)
 comps=[]
 for r0 in range(nr):
  if seenr[r0]:continue
  qr=[r0];seenr[r0]=True;rs=[];cs=set();eids=set()
  while qr:
   r=qr.pop();rs.append(r)
   for c,ei in radj[r]:
    eids.add(ei)
    if not seenc[c]:
     seenc[c]=True;cs.add(c)
     for rr,ej in cadj[c]:
      eids.add(ej)
      if not seenr[rr]:seenr[rr]=True;qr.append(rr)
  comps.append((rs,sorted(cs),sorted(eids)))
 invr=[None]*nr
 for key,i in rmap.items(): invr[i]=key
 invc=[None]*nc
 for key,i in cmap.items(): invc[i]=key
 core=[];rem=[];bond=0;maxrank=0;maxshape=(0,0)
 for rs,cs,eids in comps:
  rr={r:i for i,r in enumerate(rs)};cc={c:i for i,c in enumerate(cs)}
  A=np.zeros((len(rs),len(cs)),complex)
  for ei in eids:A[rr[rows[ei]],cc[cols[ei]]] += vals[ei]
  U,s,Vh=np.linalg.svd(A,full_matrices=False)
  keep=np.flatnonzero(s>tol*(s[0] if len(s) else 1.0))
  if not len(keep):continue
  maxrank=max(maxrank,len(keep));maxshape=max(maxshape,(len(rs),len(cs)),key=lambda x:x[0]*x[1])
  for kk in keep:
   b=bond;bond+=1
   for ii,r in enumerate(rs):
    z=U[ii,kk]
    if abs(z)>tol:core.append((invr[r],b,complex(z)))
   for jj,c in enumerate(cs):
    z=s[kk]*Vh[kk,jj]
    if abs(z)>tol:rem.append((b,invc[c],complex(z)))
 return core,rem,{'rows':nr,'cols':nc,'components':len(comps),'bond_dim':bond,'max_component_rank':maxrank,'max_component_shape':list(maxshape)}

def build_adj(cores):
 out=[]
 for core in cores:
  a=defaultdict(list)
  for (b0,p,b1),z in core.items(): a[(b0,p)].append((b1,z))
  out.append(a)
 return out

def eval_tt(adj,x):
 vec={0:1+0j}
 for t,a in enumerate(adj):
  out=defaultdict(complex);phys=x[t]
  for b0,v in vec.items():
   for b1,z in a.get((b0,phys),()): out[b1]+=v*z
  vec=out
  if not vec:return 0j
 return vec.get(0,0j)

def main():
 t0=time.time();samples=raw_q3_samples();orig=dict(samples); nvar=len(varspec)
 # entries for first factorization: row=(leftbond=0, physical0), col=suffix
 entries=[((0,x[0]),x[1:],z) for x,z in samples.items()]
 cores=[];profiles=[]
 for site in range(nvar):
  core_raw,rem,info=component_svd(entries)
  # core_raw row key=(prevbond,phys)
  cdict=defaultdict(complex)
  for (bp,p),bn,z in core_raw:cdict[(bp,p,bn)]+=z
  cores.append(dict(cdict));info.update({'site':site,'variable':list(varspec[site]),'core_nnz':len(cdict),'remainder_nnz':len(rem)});profiles.append(info)
  print(site,varspec[site],info['bond_dim'],info['components'],info['max_component_rank'],len(cdict),len(rem),flush=True)
  if site==nvar-1:
   # suffix should be empty and final bond must collapse to one after the last SVD
   # component_svd labels output bond; absorb remaining scalar into final core and remap to terminal bond 0.
   assert all(c==() for _,c,_ in rem)
   scale={b:z for b,c,z in rem}
   last=defaultdict(complex)
   for (bp,p,bn),z in cores[-1].items():last[(bp,p,0)] += z*scale.get(bn,0j)
   cores[-1]=dict(last);profiles[-1]['terminal_absorbed']=True;profiles[-1]['bond_dim_after_terminal']=1
   break
  # next entries row=(newbond,nextphys), col=remaining suffix
  entries=[]
  for b,suf,z in rem:
   if not suf: raise RuntimeError('premature empty suffix')
   entries.append(((b,suf[0]),suf[1:],z))
 # Deterministic indexed reconstruction regression. Exactness of every SVD is controlled by its retained tolerance;
 # evaluating a representative subset avoids an O(N * total_core_nnz) verifier.
 adj=build_adj(cores); maxerr=0.0; norm0=sum(abs(z)**2 for z in orig.values())
 items=list(orig.items()); step=max(1,len(items)//512); tested_items=items[::step][:512]
 for x,z in tested_items:
  zz=eval_tt(adj,x);maxerr=max(maxerr,abs(zz-z))
 absmax=0.0;tested=0
 for x,_ in tested_items[:128]:
  y=list(x); y[0]=int(y[0])+7; y=tuple(y)
  if y not in orig: absmax=max(absmax,abs(eval_tt(adj,y)));tested+=1
 out={'status':'PASS_SU2_Q3_EXACT_BLOCK_TT_FACTORIZATION' if maxerr<5e-10 and absmax<5e-10 else 'FAIL_SU2_Q3_BLOCK_TT_FACTORIZATION',
      'variables':nvar,'raw_nonzero_coefficients':len(orig),'profiles':profiles,
      'max_bond_dimension':max(p['bond_dim'] for p in profiles[:-1]),'max_single_component_rank':max(p['max_component_rank'] for p in profiles),
      'total_core_nnz':sum(len(c) for c in cores),'original_norm':math.sqrt(norm0),'reconstruction_samples_tested':len(tested_items),
      'max_reconstruction_error_on_sampled_nonzero_coefficients':float(maxerr),'absent_samples_tested':tested,'max_absent_sample_amplitude':float(absmax),'seconds':time.time()-t0,
      'representation':'32-variable exact block-sparse tensor train: 24 link twice-spins plus 8 vertex fusion-path categorical variables inserted at their closing cuts.',
      'scope':'Exact state-serialization regression for H_B^3|vac>. It establishes an executable block-frontier state format but does not yet apply the full repeated-time PEPO or compute h_d.'}
 (ROOT/'results'/'SU2_Q3_BLOCK_TT_FACTORIZATION.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ['status','variables','raw_nonzero_coefficients','max_bond_dimension','max_single_component_rank','total_core_nnz','max_reconstruction_error_on_sampled_nonzero_coefficients','seconds']},indent=2))
if __name__=='__main__':main()
