from __future__ import annotations
import numpy as np,itertools,importlib.util,time,json,hashlib,argparse
from pathlib import Path
BASE=Path('/mnt/data/task2_continue');ap=argparse.ArgumentParser();ap.add_argument('--Sold',type=int,required=True);ap.add_argument('--Snew',type=int,required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
spec=importlib.util.spec_from_file_location('g',str(BASE/'u1_l2_gauss_kato_galerkin.py'));g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);links=g.links;li=g.li
PERMS=list(itertools.permutations(range(3)));TRANS=list(itertools.product(range(2),repeat=3));MAPS=[]
for t in TRANS:
 for perm in PERMS:
  no=np.empty(24,dtype=np.int64)
  for old,(mu,x,y,z) in enumerate(links):
   c=[(x+t[0])%2,(y+t[1])%2,(z+t[2])%2];cp=[c[perm.index(i)] for i in range(3)];no[old]=li[(perm[mu],*cp)]
  inv=np.empty(24,dtype=np.int64);inv[no]=np.arange(24);MAPS.append(inv)
BASECODE=13;SHIFT=6;pow12=(BASECODE**np.arange(11,-1,-1,dtype=np.uint64))
def codes(A):return (A[:,:12].astype(np.int64)+SHIFT).astype(np.uint64)@pow12,(A[:,12:].astype(np.int64)+SHIFT).astype(np.uint64)@pow12
def canon(A):
 n=len(A);b1=np.full(n,np.iinfo(np.uint64).max,dtype=np.uint64);b2=b1.copy();br=np.empty_like(A)
 for inv in MAPS:
  X=A[:,inv];c1,c2=codes(X);m=(c1<b1)|((c1==b1)&(c2<b2));b1[m]=c1[m];b2[m]=c2[m];br[m]=X[m]
 return b1,b2,br
def load_reps(path):
 z=np.load(path);return z['reps'].astype(np.int16),z['sizes'].astype(np.int16),z['v'] if 'v' in z.files else None,z['phase'] if 'phase' in z.files else None
reps,sizes,v,phase=load_reps(a.input); c1,c2,_=canon(reps);known={(int(x),int(y)) for x,y in zip(c1,c2)};t0=time.time();print('start',len(reps),'known',len(known),flush=True)
front=reps; new_total=[];iters=[]
for it in range(20):
 found={}
 for bs in range(0,len(front),500):
  B=front[bs:bs+500];cand=[]
  for p in g.plaquettes:
   for sg in (-1,1):
    F=B+sg*p; e2=np.sum(F.astype(np.int32)**2,axis=1);m=(e2>a.Sold)&(e2<=a.Snew)
    if np.any(m):cand.append(F[m])
  if not cand:continue
  C=np.vstack(cand);x,y,R=canon(C)
  for xx,yy,rr in zip(x,y,R):
   k=(int(xx),int(yy))
   if k not in known and k not in found:found[k]=rr.copy()
 if not found:break
 new=np.asarray(list(found.values()),dtype=np.int16);new_total.append(new);known.update(found.keys());front=new;iters.append(len(new));print('iter',it+1,'new',len(new),'total',len(known),'sec',time.time()-t0,flush=True)
# combine reps plus new in canonical row form; recanonicalize all and sort by key
allr=np.vstack([reps]+new_total) if new_total else reps.copy();x,y,cr=canon(allr);order=np.lexsort((y,x));x=x[order];y=y[order];cr=cr[order];assert len(set(zip(map(int,x),map(int,y))))==len(cr)
# orbit sizes
os=[]
for rr in cr:
 imgs=np.asarray([rr[inv] for inv in MAPS],dtype=np.int16);os.append(len(np.unique(imgs,axis=0)))
os=np.asarray(os,dtype=np.int16)
np.savez_compressed(a.output,reps=cr,sizes=os,key1=x,key2=y)
out={'status':f'PASS_ORBIT_BASIS_S{a.Snew}','Sold':a.Sold,'Snew':a.Snew,'old_orbits':len(reps),'new_iterations':iters,'total_orbits':len(cr),'max_orbit_size':int(os.max()),'seconds':time.time()-t0};q=BASE/f'U1_ORBIT_BASIS_S{a.Snew}.json';q.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));print('sha256',hashlib.sha256(q.read_bytes()).hexdigest())
