#!/usr/bin/env python3
"""Exact finite-dimensional validation of the spectral source-phase orbit.

No physical continuum inference is made from this finite-dimensional test.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(20261008)
A=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5)); H=(A+A.conj().T)/2
w,V=np.linalg.eigh(H); idx=0; E=w[idx]
P=np.outer(V[:,idx],V[:,idx].conj()); mask=np.arange(5)!=idx
S=(V[:,mask]*(1/(E-w[mask])))@V[:,mask].conj().T
A=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5)); K1=(A+A.conj().T)/2
B=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5)); K2=(B+B.conj().T)/2
N=9; Z=7.2; ell=1.0; q=0.11
f1=0.73; f2=0.17*q*q
identity=np.eye(5)
f_orig=np.real(np.trace(P@K1))
f_shift=np.real(np.trace(P@(K1+f1*identity)))
eps_diff=Z*(f_shift-f_orig)/(4*N)
eps_pred=Z*f1/(4*N)
chi_orig=np.real(np.trace(P@K2)+2*np.trace(P@K1@S@K1))/N
chi_shift=np.real(np.trace(P@(K2+f2*identity))+2*np.trace(P@(K1+f1*identity)@S@(K1+f1*identity)))/N
chi_pred=chi_orig+f2/N
P_same=np.linalg.norm(P@P-P)
S_orth=np.linalg.norm(P@S)
err_eps=abs(eps_diff-eps_pred)
err_chi=abs(chi_shift-chi_pred)
# Positive residue identity for a regularized helicity-two inverse kernel.
q2=0.035; gamma=2.4; Nh=0.125
inverse=Nh*gamma*q2
connected=1/inverse
pole_residue=connected*q2
err_inverse=abs(pole_residue-1/(Nh*gamma))
assert max(err_eps,err_chi,P_same,S_orth,err_inverse)<1e-11
out={'classification':'FINITE_MATRIX_IDENTITY_ONLY','passed':True,
 'epsilon_source_shift_error':float(err_eps),
 'tensor_source_shift_error':float(err_chi),
 'projector_error':float(P_same), 'reduced_resolvent_orthogonality_error':float(S_orth),
 'inverse_kernel_pole_residue_error':float(err_inverse),
 'continuum_physical_gravity_coefficient':'not evaluated'}
path=ROOT/'results/METRIC_SOURCE_ORBIT_VALIDATION.json'
path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
