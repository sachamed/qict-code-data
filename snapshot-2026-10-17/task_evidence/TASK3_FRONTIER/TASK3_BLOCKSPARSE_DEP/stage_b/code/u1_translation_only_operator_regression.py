#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math, time
from pathlib import Path
import numpy as np
from scipy.special import jv

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'results'
PHI=math.pi/63.0; S=24
z=np.load(DATA/'U1_L2_KATO_S24_TRANSLATION_ONLY.npz')
reps=np.asarray(z['reps'],dtype=np.int8); sizes=np.asarray(z['sizes'],dtype=np.float64); v=np.asarray(z['v'],dtype=np.complex128)
N=len(reps)
L=2; sites=[(x,y,zz) for x in range(2) for y in range(2) for zz in range(2)]
links=[(mu,*s) for mu in range(3) for s in sites]; li={e:i for i,e in enumerate(links)}
plaquettes=[]
for mu in range(3):
  for nu in range(mu+1,3):
    for s0 in sites:
      s=list(s0); sm=s.copy(); sm[mu]^=1; sn=s.copy(); sn[nu]^=1
      p=np.zeros(24,dtype=np.int8)
      for e,sgn in [((mu,*s),1),((nu,*sm),1),((mu,*sn),-1),((nu,*s),-1)]: p[li[e]]+=sgn
      plaquettes.append(p)
TRANS=list(itertools.product(range(2),repeat=3)); TINV=[]
for t in TRANS:
  no=np.empty(24,dtype=np.int64)
  for old,(mu,x,y,zz) in enumerate(links): no[old]=li[(mu,(x+t[0])%2,(y+t[1])%2,(zz+t[2])%2)]
  inv=np.empty(24,dtype=np.int64); inv[no]=np.arange(24); TINV.append(inv)
BC=9; SHIFT=4; pow12=(BC**np.arange(11,-1,-1,dtype=np.uint64))
def codes(A):
  A=A.astype(np.int16,copy=False)
  return ((A[:,:12]+SHIFT).astype(np.uint64)@pow12,(A[:,12:]+SHIFT).astype(np.uint64)@pow12)
def canon_codes(A):
  n=len(A); ba=np.full(n,np.iinfo(np.uint64).max,dtype=np.uint64); bb=ba.copy()
  for inv in TINV:
    a,b=codes(A[:,inv]); m=(a<ba)|((a==ba)&(b<bb)); ba[m]=a[m];bb[m]=b[m]
  return ba,bb
ka,kb=codes(reps); keys=np.empty(N,dtype=[('a','<u8'),('b','<u8')]);keys['a']=ka;keys['b']=kb
assert np.all((ka[1:]>ka[:-1])|((ka[1:]==ka[:-1])&(kb[1:]>kb[:-1])))

def build_targets():
  T=np.full((48,N),N,dtype=np.uint32); row=0; t0=time.time(); missing=0
  e2=np.sum(reps.astype(np.int16)**2,axis=1)
  for p in plaquettes:
    for sg in (-1,1):
      F=reps+sg*p; good=(np.sum(F.astype(np.int16)**2,axis=1)<=S); ids=np.flatnonzero(good)
      a,b=canon_codes(F[good]);q=np.empty(len(a),dtype=keys.dtype);q['a']=a;q['b']=b
      j=np.searchsorted(keys,q); ok=(j<N)&(keys[np.minimum(j,N-1)]==q)
      if not np.all(ok): missing+=int(np.sum(~ok))
      T[row,ids[ok]]=j[ok].astype(np.uint32);row+=1
  return T,time.time()-t0,missing

def kb_apply(x,T):
  y=96.0*np.asarray(x,dtype=np.complex128)
  for r in range(48):
    t=T[r]; m=(t<N); tt=t[m].astype(np.int64)
    y[m] += -2.0*np.sqrt(sizes[m]/sizes[tt])*x[tt]
  return y

def ub_apply(x,T,adj=False):
  center=rad=96.0; aa=PHI*rad
  for deg in range(100):
    tail=2*sum(abs(jv(k,aa)) for k in range(deg+1,220))
    if tail<2e-13: break
  ph=np.exp(-1j*PHI*center); co=[jv(0,aa)]+[2*((-1j)**k)*jv(k,aa) for k in range(1,deg+1)]
  if adj: co=[np.conj(c) for c in co]; ph=np.conj(ph)
  def X(a): return (kb_apply(a,T)-center*a)/rad
  t0=np.asarray(x,complex); y=co[0]*t0
  if deg:
    t1=X(t0); y+=co[1]*t1
    for kk in range(2,deg+1):
      t2=2*X(t1)-t0; y+=co[kk]*t2; t0,t1=t1,t2
  return ph*y,deg,tail

def main():
  t0=time.time();T,buildsec,missing=build_targets()
  # Hermiticity randomized bilinear test.
  rng=np.random.default_rng(210919); herm=0.0
  for _ in range(3):
    a=rng.normal(size=N)+1j*rng.normal(size=N);b=rng.normal(size=N)+1j*rng.normal(size=N)
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    lhs=np.vdot(a,kb_apply(b,T));rhs=np.vdot(kb_apply(a,T),b);herm=max(herm,float(abs(lhs-rhs)))
  KE=np.sum(2*reps.astype(np.int32)**2+reps.astype(np.int32),axis=1).astype(float)
  ue=np.exp(-1j*PHI*KE)
  uv,deg,tail=ub_apply(ue*v,T,False); lam=np.vdot(v,uv); res=float(np.linalg.norm(uv-lam*v))
  out={
    'status':'PASS_U1_TRANSLATION_ONLY_NEUTRAL_OPERATOR' if missing==0 and herm<5e-12 and res<5e-11 else 'CHECK_U1_TRANSLATION_ONLY_NEUTRAL_OPERATOR',
    'S':S,'translation_orbit_dim':N,'transition_slots':int(T.size),'valid_transition_slots':int(np.sum(T<N)),
    'missing_target_states':missing,'target_table_build_seconds':buildsec,'randomized_hermiticity_residual':herm,
    'chebyshev_degree':deg,'chebyshev_tail_bound':tail,'kato_rayleigh':[float(lam.real),float(lam.imag)],
    'kato_modulus':float(abs(lam)),'kato_eigen_residual':res,'total_seconds':time.time()-t0,
    'scope':'Exact S=24 translation-only lift regression of the neutral U(1) Kato operator. It removes the cubic-permutation quotient required by ordered C8, but S=24 is an auxiliary Galerkin diagnostic and is not the final Peter-Weyl PEPO regulator.'
  }
  (OUT/'U1_TRANSLATION_ONLY_NEUTRAL_OPERATOR.json').write_text(json.dumps(out,indent=2)+'\n')
  print(json.dumps(out,indent=2))
if __name__=='__main__':main()
