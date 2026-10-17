#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'results'
z=np.load(DATA/'U1_L2_KATO_S24_STATE.npz')
reps=np.asarray(z['reps'],dtype=np.int16); sizes_full=np.asarray(z['sizes'],dtype=np.int16); v=np.asarray(z['v'],dtype=np.complex128)
L=2; links=[(mu,x,y,zz) for mu in range(3) for x in range(L) for y in range(L) for zz in range(L)]; li={t:i for i,t in enumerate(links)}
TRANS=list(itertools.product(range(2),repeat=3)); PERMS=list(itertools.permutations(range(3)))
TINV=[]
for t in TRANS:
 no=np.empty(24,dtype=np.int64)
 for old,(mu,x,y,zz) in enumerate(links): no[old]=li[(mu,(x+t[0])%2,(y+t[1])%2,(zz+t[2])%2)]
 inv=np.empty(24,dtype=np.int64); inv[no]=np.arange(24); TINV.append(inv)
PINV=[]
for perm in PERMS:
 no=np.empty(24,dtype=np.int64); invp=[perm.index(i) for i in range(3)]
 for old,(mu,x,y,zz) in enumerate(links):
  c=(x,y,zz); cp=[c[invp[i]] for i in range(3)]; no[old]=li[(perm[mu],*cp)]
 inv=np.empty(24,dtype=np.int64);inv[no]=np.arange(24);PINV.append(inv)
BC=13;SHIFT=6;pow12=(BC**np.arange(11,-1,-1,dtype=np.uint64))
def codes(A):
 return ((A[:,:12].astype(np.int64)+SHIFT).astype(np.uint64)@pow12,
         (A[:,12:].astype(np.int64)+SHIFT).astype(np.uint64)@pow12)
def tcanon_codes(A):
 n=len(A); best_a=np.full(n,np.iinfo(np.uint64).max,dtype=np.uint64); best_b=best_a.copy()
 for inv in TINV:
  a,b=codes(A[:,inv]); m=(a<best_a)|((a==best_a)&(b<best_b)); best_a[m]=a[m];best_b[m]=b[m]
 return best_a,best_b
N=len(reps); all_a=[];all_b=[];all_o=[];t0=time.time()
for inv in PINV:
 a,b=tcanon_codes(reps[:,inv]);all_a.append(a);all_b.append(b);all_o.append(np.arange(N,dtype=np.int32))
a=np.concatenate(all_a);b=np.concatenate(all_b);o=np.concatenate(all_o)
ord=np.lexsort((b,a));a=a[ord];b=b[ord];o=o[ord]
new=np.ones(len(a),dtype=bool);new[1:]=(a[1:]!=a[:-1])|(b[1:]!=b[:-1])
# Check duplicate owners agree.
start=np.flatnonzero(new); end=np.r_[start[1:],len(a)]; conflicts=0
for s,e in zip(start,end):
 if e-s>1 and np.any(o[s:e]!=o[s]): conflicts+=1
assert conflicts==0
ua=a[new];ub=b[new];own=o[new]; M=len(ua)
# Decode base-13 code pairs to representatives.
def decode_half(c):
 c=c.copy(); A=np.empty((len(c),12),dtype=np.int16)
 for j in range(11,-1,-1): A[:,j]=(c%BC).astype(np.int16)-SHIFT;c//=BC
 return A
tr=np.hstack([decode_half(ua),decode_half(ub)])
# translation orbit size via stabilizer count
stab=np.zeros(M,dtype=np.int8)
for inv in TINV: stab += np.all(tr[:,inv]==tr,axis=1).astype(np.int8)
ts=(8//stab).astype(np.int16)
parts=np.zeros(N,dtype=np.int16);weights=np.zeros(N,dtype=np.int16)
np.add.at(parts,own,1);np.add.at(weights,own,ts)
assert np.array_equal(weights,sizes_full)
vt=v[own]*np.sqrt(ts.astype(float)/sizes_full[own].astype(float))
vback=np.zeros_like(v);np.add.at(vback,own,vt*np.sqrt(ts.astype(float)/sizes_full[own].astype(float)))
out={
 'status':'PASS_U1_TRANSLATION_ONLY_SYMMETRY_LIFT','input_full_symmetry_orbit_dim':int(N),'translation_only_orbit_dim':int(M),
 'dimension_increase_factor':float(M/N),'full_flux_basis_dim':int(np.sum(sizes_full)),
 'translation_orbit_size_histogram':{str(k):int(np.sum(ts==k)) for k in sorted(set(map(int,ts)))},
 'full_orbit_translation_suborbit_count_histogram':{str(k):int(np.sum(parts==k)) for k in sorted(set(map(int,parts)))},
 'max_suborbits_per_full_orbit':int(parts.max()),'orbit_partition_exact':True,
 'input_vector_norm':float(np.linalg.norm(v)),'lifted_vector_norm':float(np.linalg.norm(vt)),
 'norm_residual':float(abs(np.linalg.norm(vt)-np.linalg.norm(v))),'roundtrip_projection_residual':float(np.linalg.norm(vback-v)),
 'seconds':time.time()-t0,
 'scientific_scope':'Exact lift of the existing S=24 U(1) Kato regression from translation x cubic-permutation orbits to translation-only orbits. It removes the axis-permutation quotient before ordered C8 diagnostics. S=24 remains an auxiliary Galerkin regression, not the final PEPO regulator.'}
np.savez_compressed(DATA/'U1_L2_KATO_S24_TRANSLATION_ONLY.npz',reps=tr,sizes=ts,owner=own,v=vt)
(OUT/'U1_TRANSLATION_ONLY_LIFT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
