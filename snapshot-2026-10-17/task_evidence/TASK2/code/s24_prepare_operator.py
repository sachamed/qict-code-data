import numpy as np,itertools,importlib.util,time,math,json,hashlib
from pathlib import Path
from scipy.sparse import coo_matrix,identity
BASE=Path('/mnt/data/task2_continue');S=24
spec=importlib.util.spec_from_file_location('g',str(BASE/'u1_l2_gauss_kato_galerkin.py'));g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);links=g.links;li=g.li
PERMS=list(itertools.permutations(range(3)));TRANS=list(itertools.product(range(2),repeat=3));MAPS=[]
for t in TRANS:
 for perm in PERMS:
  no=np.empty(24,dtype=np.int64)
  for old,(mu,x,y,z) in enumerate(links):
   c=[(x+t[0])%2,(y+t[1])%2,(z+t[2])%2];cp=[c[perm.index(i)] for i in range(3)];no[old]=li[(perm[mu],*cp)]
  inv=np.empty(24,dtype=np.int64);inv[no]=np.arange(24);MAPS.append(inv)
BC=13;SHIFT=6;pow12=(BC**np.arange(11,-1,-1,dtype=np.uint64))
def codes(A):return (A[:,:12].astype(np.int64)+SHIFT).astype(np.uint64)@pow12,(A[:,12:].astype(np.int64)+SHIFT).astype(np.uint64)@pow12
def st(x,y):
 z=np.empty(len(x),dtype=[('a','<u8'),('b','<u8')]);z['a']=x;z['b']=y;return z
def canon(A):
 n=len(A);x=np.full(n,np.iinfo(np.uint64).max,dtype=np.uint64);y=x.copy()
 for inv in MAPS:
  B=A[:,inv];a,b=codes(B);m=(a<x)|((a==x)&(b<y));x[m]=a[m];y[m]=b[m]
 return st(x,y)
z=np.load(BASE/'u1_orbits_S24.npz');reps=z['reps'].astype(np.int16);sizes=z['sizes'].astype(float);keys=canon(reps);n=len(reps);RR=[];CC=[];t0=time.time()
for bs in range(0,n,1000):
 B=reps[bs:bs+1000];src=np.arange(bs,min(bs+1000,n),dtype=np.int32);cand=[];sr=[]
 for p in g.plaquettes:
  for sg in (-1,1):
   F=B+sg*p;good=np.sum(F.astype(np.int32)**2,axis=1)<=S
   if np.any(good):cand.append(F[good]);sr.append(src[good])
 C=np.vstack(cand);ss=np.concatenate(sr);q=canon(C);j=np.searchsorted(keys,q);assert np.all((j<n)&(keys[np.minimum(j,n-1)]==q));RR.append(ss);CC.append(j.astype(np.int32))
rr=np.concatenate(RR);cc=np.concatenate(CC);dat=-2*np.sqrt(sizes[rr]/sizes[cc]);KB=coo_matrix((dat,(rr,cc)),shape=(n,n)).tocsr();KB.sum_duplicates();KB=KB+96*identity(n,format='csr');D=KB-KB.T;herm=0 if D.nnz==0 else float(np.max(np.abs(D.data)));KB=(KB+KB.T)*0.5;KB.eliminate_zeros();KE=np.asarray([sum(2*int(e)*int(e)+int(e) for e in r) for r in reps],float)
# embed prev
zp=np.load(BASE/'u1_orbits_S22_kato.npz');pkeys=canon(zp['reps'].astype(np.int16));pos=np.searchsorted(keys,pkeys);assert np.all((pos<n)&(keys[pos]==pkeys));v0=np.zeros(n,complex);v0[pos]=zp['v'];v0/=np.linalg.norm(v0);th=float(zp['phase'][0])
np.savez_compressed(BASE/'S24_OPERATOR.npz',reps=reps,sizes=sizes.astype(np.int16),KE=KE,KB_data=KB.data,KB_indices=KB.indices,KB_indptr=KB.indptr,v0=v0,prev_phase=np.array([th]))
out={'status':'PASS_S24_OPERATOR_PREPARED','dim':n,'KB_nnz':int(KB.nnz),'hermiticity':herm,'prev_phase':th,'seconds':time.time()-t0};p=BASE/'S24_OPERATOR.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));print(hashlib.sha256(p.read_bytes()).hexdigest())
