#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
L=2
S=[(x,y,z) for x in range(2) for y in range(2) for z in range(2)]
def shift(x,mu):
 a=list(x);a[mu]^=1;return tuple(a)
def rand_su2(rng):
 q=rng.normal(size=4);q/=np.linalg.norm(q);w,x,y,z=q
 return np.array([[w+1j*z,y+1j*x],[-y+1j*x,w-1j*z]],complex)
def path_product(U,order):
 s=(0,0,0); P=np.eye(2,dtype=complex)
 for mu in order:
  P=U[(mu,s)]@P; s=shift(s,mu)
 return P,s
def gauge_transform(U,g):
 out={}
 for (mu,s),A in U.items():
  t=shift(s,mu); out[(mu,s)]=g[t]@A@g[s].conj().T
 return out
def main():
 rng=np.random.default_rng(20260919); max_path=0.; max_loop=0.; rows=[]
 for k in range(64):
  U={(mu,s):rand_su2(rng) for mu in range(3) for s in S}; g={s:rand_su2(rng) for s in S}
  UA,tA=path_product(U,(0,1,2)); UB,tB=path_product(U,(2,1,0)); assert tA==tB==(1,1,1)
  W=UB.conj().T@UA
  Ug=gauge_transform(U,g); UAg,_=path_product(Ug,(0,1,2)); UBg,_=path_product(Ug,(2,1,0)); Wg=UBg.conj().T@UAg
  expA=g[(1,1,1)]@UA@g[(0,0,0)].conj().T
  expW=g[(0,0,0)]@W@g[(0,0,0)].conj().T
  e1=np.linalg.norm(UAg-expA,2);e2=np.linalg.norm(Wg-expW,2)
  max_path=max(max_path,float(e1));max_loop=max(max_loop,float(e2));
  if k<5: rows.append({'seed_index':k,'path_covariance_residual':float(e1),'closed_loop_covariance_residual':float(e2),'closed_loop_unitarity_defect':float(np.linalg.norm(W.conj().T@W-np.eye(2),2))})
 assert max_path<2e-14 and max_loop<2e-14
 out={'status':'PASS_SU2_REFERENCE_PATHS_DIFFER_BY_CLOSED_LOOP_COORDINATE','tests':64,'max_open_path_covariance_residual':max_path,'max_closed_loop_covariance_residual':max_loop,'sample_rows':rows,
 'identity':'For two Wilson paths A,B with the same endpoints, W=B^dag A is a closed-loop holonomy based at the source. A change of reference string therefore multiplies the affine charged-sector coordinate by an element of the closed-loop algebra; it does not define a new Gauss superselection sector.',
 'task3_consequence':'In the C10-selected charged sector one reference string can be chosen for coordinates. Physical state selection is performed by Kato continuation in the complete plaquette/closed-loop sector; no probabilistic thin-string superposition is required.'}
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
