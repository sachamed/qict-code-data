#!/usr/bin/env python3
"""Independent finite-dimensional checks of phase, projective geometry and Floquet ambiguity."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
HERE=Path(__file__).resolve().parent
out={'classification':'Mathematical comparators, not the canonical seven-factor physical coframe sector','tests':[]}
def check(label, residual, tolerance):
    residual=float(residual)
    assert residual<tolerance,(label,residual,tolerance)
    out['tests'].append({'name':label,'residual':residual,'tolerance':tolerance,'passed':True})

H0=np.array([[.3,.11],[.11,.7]],dtype=complex)
Hq=np.array([[.5,-.2j],[.2j,.8]],dtype=complex)
U0=expm(-1j*H0);Uq=expm(-1j*Hq)
F=lambda g:np.sin(g)**2+g**2
lam=1.6
for g in (.2,.4,.9):
    phase=np.exp(1j*lam*F(g))
    W=np.kron(U0.conj(),Uq)
    Wg=np.kron((phase*U0).conj(),phase*Uq)
    check('common_phase_relative_matvec_g_'+str(g),np.linalg.norm(W-Wg,2),1e-13)
# Closed time path / geometrical branch contrast; select a trivial matter U in which phase can be isolated.
for gp,gm in [(.3,.2),(.8,.1),(.45,.45)]:
    # F_g+g- = Tr[rho (U[g-])^dagger U[g+]], U[g]=exp(i lambda F(g)) U0
    Fctp=np.trace((np.eye(2)/2)@(np.exp(1j*lam*F(gp))*U0)@(np.exp(1j*lam*F(gm))*U0).conj().T)
    check('branch_reference_interference_'+str((gp,gm)),abs(Fctp-np.exp(1j*lam*(F(gp)-F(gm)))),1e-13)
# Projective quantum metric is insensitive to arbitrary phase even for a nontrivial parameterized state.
psi=np.array([1.,0.],complex)
A=np.array([[.31,.16],[.16,-.22]],complex)
for theta in [.1,.3]:
    p=expm(-1j*theta*A)@psi
    dp=-1j*A@p
    dthetaF=2*theta # local phase lambda theta^2
    p2=np.exp(1j*lam*theta**2)*p
    dp2=np.exp(1j*lam*theta**2)*(dp+1j*lam*dthetaF*p)
    q=lambda state,der:np.vdot(der,der)-np.abs(np.vdot(state,der))**2
    check('projective_metric_phase_invariance_theta_'+str(theta),abs(q(p,dp)-q(p2,dp2)),1e-13)
# Floquet alias: even perfectly resolved eigenphase cannot determine an unwrapped energy.
tau=.1
for e in [.87,1.69,5.]:
    for n in [-4,-1,1,4]:
        check('Floquet_energy_alias_'+str((e,n)),abs(np.exp(-1j*tau*e)-np.exp(-1j*tau*(e+2*np.pi*n/tau))),1e-13)

out['test_count']=len(out['tests'])
(HERE/'phase_floquet_tests.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'test_count':out['test_count'],'max_residual':max(t['residual'] for t in out['tests'])},indent=2))
