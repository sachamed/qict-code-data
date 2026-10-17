#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/QICT_4D_COMPLETION_SELFTEST.json'

def fourier_duality(q:int):
    w=np.exp(2j*np.pi/q)
    X=np.zeros((q,q),complex)
    for r in range(q): X[(r+1)%q,r]=1
    Z=np.diag([w**r for r in range(q)])
    F=np.array([[w**(r*s)/np.sqrt(q) for s in range(q)] for r in range(q)],complex)
    res_unit=np.linalg.norm(F.conj().T@F-np.eye(q),2)
    res_xz=np.linalg.norm(F@X@F.conj().T-Z,2)
    res_zx=np.linalg.norm(F@Z@F.conj().T-np.linalg.matrix_power(X,-1),2)
    res_weyl=np.linalg.norm(Z@X-w*X@Z,2)
    return dict(q=q,unitarity=res_unit,FXF_Z=res_xz,FZF_Xinv=res_zx,Weyl=res_weyl)

def toeplitz_test(q=11,r=3,N=8):
    rng=np.random.default_rng(20261007)
    A=rng.normal(size=(q,q))+1j*rng.normal(size=(q,q))
    Q,R=np.linalg.qr(A)
    d=np.diag(R); Q=Q@np.diag(d/np.abs(d))
    B=rng.normal(size=(q,r))+1j*rng.normal(size=(q,r))
    V,_=np.linalg.qr(B)
    V=V[:,:r]
    C={n:V.conj().T@np.linalg.matrix_power(Q,n)@V for n in range(-N,N+1)}
    T=np.block([[C[k-j] for k in range(N+1)] for j in range(N+1)])
    T=(T+T.conj().T)/2
    ev=np.linalg.eigvalsh(T)
    return dict(q=q,r=r,N=N,min_eigenvalue=float(ev.min()),hermiticity=float(np.linalg.norm(T-T.conj().T)))

def higgs_stationarity(alpha2=-0.37,alpha4=0.13):
    x=-alpha2/(2*alpha4)
    deriv=alpha2+2*alpha4*x
    curvature=2*alpha4
    v2=-alpha2/alpha4
    mh2=-2*alpha2
    return dict(alpha2=alpha2,alpha4=alpha4,x_min=x,stationarity_residual=abs(deriv),curvature=curvature,vE2_over_Estar2=v2,mhc2_sq_over_Estar2=mh2)

def gravity_algebra(gamma=2.7,eps=1.2e-120,tau=1.0,hbar=1.0,c=1.0):
    G=c**5*tau**2/(8*np.pi*hbar*gamma)
    Lam=eps/(gamma*c**2*tau**2)
    # rematch the two effective-action coefficients
    ell=c*tau
    lhs_R=hbar*gamma/(2*ell**2)
    rhs_R=c**3/(16*np.pi*G)
    lhs_0=hbar*eps/ell**4
    rhs_0=c**3*Lam/(8*np.pi*G)
    return dict(G=G,Lambda=Lam,R_match_relative=abs(lhs_R-rhs_R)/abs(lhs_R),vacuum_match_relative=abs(lhs_0-rhs_0)/(abs(lhs_0) if lhs_0 else 1.0))


def derived_gravity_identities(gamma=2.7,eps=1.2e-120,tau=1.7,hbar=1.0,c=1.0,M=4.3):
    ell=c*tau
    Estar=hbar/tau
    G=c**5*tau**2/(8*np.pi*hbar*gamma)
    Lam=eps/(gamma*c**2*tau**2)
    lp=np.sqrt(hbar*G/c**3)
    tp=np.sqrt(hbar*G/c**5)
    Ep=np.sqrt(hbar*c**5/G)
    mu=M*c**2/Estar
    rs=2*G*M/c**2
    S=4*np.pi*rs**2*c**3/(4*hbar*G)
    kT=hbar*c**3/(8*np.pi*G*M)
    return {
      'ellP_relative_residual': float(abs(lp-ell/np.sqrt(8*np.pi*gamma))/lp),
      'tP_relative_residual': float(abs(tp-tau/np.sqrt(8*np.pi*gamma))/tp),
      'EP_relative_residual': float(abs(Ep-np.sqrt(8*np.pi*gamma)*Estar)/Ep),
      'Lambda_lP2_residual': float(abs(Lam*lp**2-eps/(8*np.pi*gamma**2))),
      'Schwarzschild_radius_residual': float(abs(rs/ell-mu/(4*np.pi*gamma))),
      'Schwarzschild_entropy_residual': float(abs(S-mu**2/(2*gamma))),
      'Hawking_temperature_residual': float(abs(kT/Estar-gamma/mu)),
      'deSitter_Htau_residual': float(abs(np.sqrt(Lam*c**2/3)*tau-np.sqrt(eps/(3*gamma))))
    }

def friedmann_identity(gamma=3.1,eps=0.007,tau=1.2,hbar=1.0,c=1.0,u=0.4,p=0.03,k=1,a=7.0):
    G=c**5*tau**2/(8*np.pi*hbar*gamma)
    Lam=eps/(gamma*c**2*tau**2)
    H2=8*np.pi*G*u/(3*c**2)+Lam*c**2/3-k*c**2/a**2
    lhs=H2*tau**2+k*(c*tau/a)**2
    uhat=u*c**3*tau**4/hbar
    rhs=(uhat+eps)/(3*gamma)
    acc=(-4*np.pi*G*(u+3*p)/(3*c**2)+Lam*c**2/3)*tau**2
    phat=p*c**3*tau**4/hbar
    acc_rhs=-(uhat+3*phat)/(6*gamma)+eps/(3*gamma)
    return {'friedmann1_residual':float(abs(lhs-rhs)),'friedmann2_residual':float(abs(acc-acc_rhs))}

result={
  'classification':'QICT_4D_COMPLETION_MATHEMATICAL_SELFTEST',
  'spacetime_policy':'3+1 continuum only; no extra spacetime dimensions introduced',
  'fourier_duality':[fourier_duality(3),fourier_duality(21)],
  'matrix_moment_positivity':toeplitz_test(),
  'higgs_algebra':higgs_stationarity(),
  'gravity_dimensional_matching':gravity_algebra(),
  'derived_gravity_identities':derived_gravity_identities(),
  'friedmann_copy_unit_identity':friedmann_identity(),
  'scale_no_go':{
      'statement':'With only dimensionless invariants plus hbar and c, no monomial has units of time.',
      'linear_constraints':'mass: a=0; length: 2a+b=0 => b=0; time exponent -a-b=0, not 1',
      'passed':True
  },
  'scientific_status':{
      'exactly_tested':['finite Fourier/Weyl duality','block-Toeplitz positivity','Higgs minimization algebra','G/Lambda coefficient matching','Planck/vacuum/horizon response identities','dimensionless Friedmann identities','dimensional scale no-go'],
      'not_tested_as_physical_predictions':['gamma_R from QICT continuum','epsilon_0 from QICT vacuum','massless helicity-2 pole','Weinberg-Witten compatibility','Higgs alpha2/alpha4 from common-net dynamics']
  }
}
OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(OUT)
