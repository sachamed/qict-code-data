#!/usr/bin/env python3
"""A posteriori positive-moment restoration, exact algebra plus finite-unitary tests.

This file does not calculate the unavailable coherent charged G2/U-cone inputs.
The nearest-cone eigenvalue is calculated numerically; the proof is exact,
while the finite-precision spectral positivity is checked with a safety margin.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'INEXACT_MOMENT_POSITIVITY_REPAIR.json'
EPS=np.finfo(float).eps

def norm(A): return float(np.linalg.norm(A,2))

def toeplitz(C):
    N=len(C)-1; d=C.shape[1]
    T=np.block([[C[j-i] if j>=i else C[i-j].conj().T for j in range(N+1)] for i in range(N+1)])
    return (T+T.conj().T)/2

def coefficients(N):
    u=np.sqrt(2/(N+2))*np.sin(np.arange(1,N+2)*np.pi/(N+2))
    return np.array([float(np.dot(u[n:],u[:N+1-n])) for n in range(N+1)])

def repair(C):
    N=len(C)-1;d=C.shape[1]
    if norm(C[0]-np.eye(d))>1e-12: raise ValueError('normalize C0=I explicitly')
    T=toeplitz(C)
    w=np.linalg.eigvalsh(T)
    guard=128*EPS*T.shape[0]*max(1,norm(T))
    gamma=max(0.0, -float(w[0])+guard) if w[0]<guard else 0.0
    D=C.copy()
    D[1:]/=1+gamma
    T2=toeplitz(D)
    exact_form=(T+gamma*np.eye(T.shape[0]))/(1+gamma)
    if norm(T2-exact_form)>2e-12: raise AssertionError('Haar-mix identity')
    if np.linalg.eigvalsh(T2).min()<-1e-11: raise AssertionError('failed positivity')
    return D,gamma,float(w[0]),float(np.linalg.eigvalsh(T2).min()),guard

def evaluate(label,C,U,V,noise,depth):
    C=C[:depth+1].copy()
    C[0]=np.eye(C.shape[1]);original=C.copy()
    if noise:
        C[1]+=noise*np.eye(C.shape[1])
    D,gamma,lo,lo2,guard=repair(C)
    Ns=len(C)-1
    eta=np.array([norm(C[n]-original[n]) for n in range(Ns+1)])
    eta2=np.array([norm(D[n]-original[n]) for n in range(Ns+1)])
    for n in range(1,Ns+1):
        if eta2[n]>(eta[n]+gamma)/(1+gamma)+1e-10:
            raise AssertionError('moment error transfer bound')
    if gamma>2*np.sum(eta[1:])+guard+1e-11:
        raise AssertionError('Toeplitz stability bound')
    checks=Ns+4; numerics=[];b=coefficients(Ns)
    for radius,phase in ((1.25,.7),(1.7,-.5),(2.0,.32),(5.,1.1)):
        z=radius*np.exp(1j*phase)
        g=V.conj().T@np.linalg.solve(z*np.eye(U.shape[0])-U,V)
        gd=sum(b[n]*D[n]/z**(n+1) for n in range(Ns+1))
        sig=z*np.eye(C.shape[1])-original[1]-np.linalg.inv(g)
        sighat=z*np.eye(C.shape[1])-D[1]-np.linalg.inv(gd)
        E=sum((1-b[n])/radius**(n+1) for n in range(1,Ns+1))+1/(radius**(Ns+1)*(radius-1))
        M=E+sum(b[n]*eta2[n]/radius**(n+1) for n in range(1,Ns+1))
        bound=eta2[1]+(radius+1)**2*M
        err=norm(sighat-sig);invg=norm(np.linalg.inv(gd))
        if err>bound+1e-10: raise AssertionError('positive-moment selfenergy error bound')
        if invg>radius+1+1e-10: raise AssertionError('inverse bound')
        checks+=2
        numerics.append({'abs_z':radius,'phase':phase,'actual_selfenergy_error':err,
                         'certified_analytic_error_budget':float(bound),
                         'repaired_inverse_norm':invg,'universal_inverse_bound':radius+1})
    return {'sample':label,'depth':Ns,'receiver_dimension':C.shape[1],
            'toeplitz_min_eigenvalue_before':lo,'toeplitz_min_eigenvalue_after':lo2,
            'numerical_eigensolver_guard':guard,'Haar_mixture_gamma':gamma,
            'max_input_moment_error':float(max(eta)),
            'max_repaired_moment_error':float(max(eta2)),
            'checks_passed':checks,'exterior_tests':numerics}

def main():
    rng=np.random.default_rng(261010)
    # Genuine finite-dimensional unitary compressed to a rank-six receiver:
    d=6;D=11
    Q,r=np.linalg.qr(rng.normal(size=(D,D))+1j*rng.normal(size=(D,D)))
    Q=Q@np.diag(np.diag(r)/np.abs(np.diag(r)))
    phases=np.exp(1j*np.linspace(.13,5.92,D))
    U=Q@np.diag(phases)@Q.conj().T
    V=Q[:,:d] # spectrum discrete with singular moment Toeplitz (stress test)
    Cs=np.array([V.conj().T@np.linalg.matrix_power(U,n)@V for n in range(8)])
    # Deliberately incompatible moments, not physical measurement errors.
    a=evaluate('rank_six_finite_unitary_perturbed',Cs,U,V,.021,7)
    b=evaluate('rank_six_finite_unitary_clean',Cs,U,V,0.,7)
    # Independent observed benchmark is a synthetic spectral measure; it is NOT G2.
    with np.load(ROOT/'results'/'SPECTRAL_BENCHMARK.npz',allow_pickle=False) as dat:
        poles=dat['poles'];residues=dat['residues'];B=dat['moments']
    # Build a finite unitary dilation of the atom/residue measure by Cholesky factors.
    # Residues may be singular; eigenfactorization respects their positive part.
    WW=[];LL=[]
    for pole,R in zip(poles,residues):
        lam,P=np.linalg.eigh((R+R.conj().T)/2)
        if lam.min()<-1e-11:raise AssertionError('benchmark residue not PSD')
        for j,val in enumerate(lam):
            if val>1e-14:
                LL.append(pole)
                WW.append(np.sqrt(val)*P[:,j].conj())
    Vb=np.stack(WW,axis=0)
    if norm(Vb.conj().T@Vb-np.eye(6))>1e-11:raise AssertionError('benchmark receiver not normalized')
    Ub=np.diag(np.array(LL))
    bclean=evaluate('benchmark_unperturbed',B,Ub,Vb,0.,7)
    bpert=evaluate('benchmark_stress_test',B,Ub,Vb,1.4,7)
    result={'classification':'FINITE_UNITARY_POSITIVITY_RESTORATION_CONTROL_NOT_PHYSICAL_G2',
            'mathematical_result':'Uniform Haar moment mixing enforces finite Toeplitz PSD and restores unconditional exterior self-energy inverse bounds.',
            'samples':[a,b,bclean,bpert], 'number_cases':4,
            'total_checks_passed':sum(x['checks_passed'] for x in (a,b,bclean,bpert)) + len(poles) + 1,
            'physical_G2_completed':False,
            'scope':'Finite matrix moment reconstruction and certified algebraic error bounds; floating eigenvalue checks are numerical, not interval arithmetic.'}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS',result['total_checks_passed'],'checks; cases',result['number_cases'])

if __name__=='__main__':main()
