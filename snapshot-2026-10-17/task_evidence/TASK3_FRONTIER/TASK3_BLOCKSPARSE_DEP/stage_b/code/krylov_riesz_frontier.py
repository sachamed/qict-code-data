#!/usr/bin/env python3
"""Source-driven Krylov/Riesz reduction for Task-3 PEPO moments.

Core observation: Task 3 needs only a fixed Kato branch and the three source
moments d=8,9,10.  A full gauge-Hilbert diagonalization is unnecessary.  The
vacuum can be selected by a Riesz spectral projector applied to the primitive
source, and the route moments are exactly determined by the source Krylov
subspace through degree d.  In the PEPO implementation each matvec is a
symmetry-block frontier sweep; its certified truncation error is accumulated.

This file validates the mathematical reduction on deterministic finite unitary
models.  The physical PEPO matvec is supplied by the Stage-B recoupling engine.
"""
from __future__ import annotations
import json, math
import numpy as np


def arnoldi(U,v,m,tol=1e-14):
    v=np.asarray(v,complex); v=v/np.linalg.norm(v)
    n=len(v); Q=np.zeros((n,m+1),complex); H=np.zeros((m+1,m),complex); Q[:,0]=v
    k=m
    for j in range(m):
        w=U@Q[:,j]
        for i in range(j+1):
            H[i,j]=np.vdot(Q[:,i],w); w-=H[i,j]*Q[:,i]
        H[j+1,j]=np.linalg.norm(w)
        if H[j+1,j]<tol:
            k=j+1; break
        Q[:,j+1]=w/H[j+1,j]
    return Q[:,:k],H[:k,:k]

def krylov_moments(U,v,dmax):
    Q,H=arnoldi(U,v,dmax+1)
    e=np.zeros(H.shape[0],complex);e[0]=1
    return [np.vdot(e,np.linalg.matrix_power(H,d)@e) for d in range(dmax+1)]

def exact_moments(U,v,dmax):
    v=np.asarray(v,complex);v=v/np.linalg.norm(v);w=v.copy(); out=[]
    for _ in range(dmax+1):
        out.append(np.vdot(v,w)); w=U@w
    return out

def riesz_projector_discrete(U,center_phase,half_width,nq=256):
    """Contour quadrature projector around an isolated unit-circle eigenphase.

    A circle centered at lambda0 with radius rho is used.  The returned matrix
    is for validation; PEPO execution applies resolvents to a source vector.
    """
    lam0=np.exp(1j*center_phase); rho=2*math.sin(half_width/2)*0.45
    n=U.shape[0]; P=np.zeros_like(U,dtype=complex); I=np.eye(n,dtype=complex)
    for k in range(nq):
        th=2*math.pi*(k+0.5)/nq; z=lam0+rho*np.exp(1j*th); dz=1j*rho*np.exp(1j*th)
        P += np.linalg.solve(z*I-U,I)*dz
    P *= (2*math.pi/nq)/(2j*math.pi)
    return P

def self_test():
    rng=np.random.default_rng(20260918)
    n=31
    # Deterministic unitary with a protected eigenphase near 0.37.
    X=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); Q,_=np.linalg.qr(X)
    phases=np.linspace(-2.8,2.8,n); phases[7]=0.37
    U=Q@np.diag(np.exp(1j*phases))@Q.conj().T
    v=rng.normal(size=n)+1j*rng.normal(size=n)
    dmax=10
    a=exact_moments(U,v,dmax); b=krylov_moments(U,v,dmax)
    moment_err=float(max(abs(x-y) for x,y in zip(a,b)))
    assert moment_err<2e-12
    gap=min(abs(((phases-phases[7]+math.pi)%(2*math.pi))-math.pi)[np.arange(n)!=7])
    P=riesz_projector_discrete(U,phases[7],min(0.15,gap*0.8),nq=512)
    # Compare to exact rank-one projector.
    q=Q[:,7]; P0=np.outer(q,q.conj()); perr=float(np.linalg.norm(P-P0,2))
    assert perr<2e-10
    pv=P@v; pv/=np.linalg.norm(pv)
    eigres=float(np.linalg.norm(U@pv-np.exp(1j*phases[7])*pv))
    assert eigres<3e-10
    return {
      'status':'PASS_SOURCE_DRIVEN_RIESZ_KRYLOV_REDUCTION',
      'tested_dimension':n,
      'route_degree_max':dmax,
      'max_moment_residual':moment_err,
      'riesz_projector_2norm_residual':perr,
      'selected_branch_eigen_residual':eigres,
      'theorem':'For a fixed source v, moments <v,U^d v> for d<=D depend only on K_D(U,v)=span{v,Uv,...,U^D v}; an isolated Kato branch is obtainable by a Riesz projector applied to the primitive source, so no full Hilbert-space diagonalization is required.',
      'pepo_realization':'Replace each U or resolvent matvec by a point-split symmetry-block PEPO frontier sweep and add its discarded-norm certificate to the moment error ledger.',
      'scope':'Algorithmic reduction validated exactly on finite unitary models; physical 24-link PEPO matvec still must be executed.'
    }

if __name__=='__main__': print(json.dumps(self_test(),indent=2))
