#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

def projector_action(A,B,center,radius,nquad,rtol,maxiter):
    th=2*np.pi*(np.arange(nquad)+0.5)/nquad; nodes=center+radius*np.exp(1j*th)
    B=np.asarray(B,np.complex128); B=B if B.ndim==2 else B[:,None]
    out=np.zeros_like(B); worst=0.0; count=0
    for z in nodes:
      M=LinearOperator(A.shape,matvec=lambda x,z=z:z*x-A.matvec(x),dtype=np.complex128)
      X=np.empty_like(B)
      for j in range(B.shape[1]):
        x,info=gmres(M,B[:,j],rtol=rtol,atol=0,maxiter=maxiter,restart=min(80,A.shape[0]))
        if info!=0: raise RuntimeError(info)
        r=B[:,j]-M.matvec(x); worst=max(worst,float(np.linalg.norm(r))); count+=1; X[:,j]=x
      out += ((z-center)/nquad)*X
    return out,count,worst

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    diag=np.array([1,1,-1,1j,-1j,np.exp(1j*np.pi/3)],complex)
    A=LinearOperator((6,6),matvec=lambda x:diag*x,dtype=np.complex128); B=np.eye(6,dtype=complex)
    Y,count,worst=projector_action(A,B,1+0j,.1,128,1e-13,100)
    exact=np.diag([1,1,0,0,0,0]).astype(complex); pe=float(np.linalg.norm(Y-exact,'fro'))
    U,s,_=np.linalg.svd(Y,full_matrices=False); keep=s>1e-13*s[0]; Q=U[:,keep]; F=Q/math.sqrt(Q.shape[1]); rho=F@F.conj().T
    re=float(np.linalg.norm(rho-.5*exact,'fro'))
    result={'calculation':'Deterministic matrix-free Riesz contour validation','quadrature_nodes':128,'linear_solves':count,'uniform_contour_gap':0.1,'projector_fro_error':pe,'density_fro_error':re,'worst_linear_residual':worst,'retained_rank':int(keep.sum())}
    Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
