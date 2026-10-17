#!/usr/bin/env python3
"""Finite algebraic regression tests for Supplementary S121.

These tests do not validate physical continuum closure or Higgs/RG coefficients.
"""
import json
from pathlib import Path
import numpy as np

checks=[]

def ck(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)

# Z6 matter kernel: y=6Y, and SU3/SU2 representation switches.
for k in range(6):
    phase=np.exp(1j*np.pi*k/3)
    center3=np.exp(2j*np.pi*k/3)
    center2=(-1)**k
    for name,c3,c2,y in [('Q_L',True,True,1),('u_R',True,False,4),
                          ('d_R',True,False,-2),('L_L',False,True,-3),
                          ('e_R',False,False,-6),('nu_R',False,False,0)]:
        action=(center3 if c3 else 1)*(center2 if c2 else 1)*phase**y
        ck(f'Z6 central kernel k={k}, rep={name}',abs(action-1)<5e-14)

# Symmetry constrains block multiplicities, not invariant eigenvalues.
P=np.diag([1,1,0,0]).astype(complex)
Q=np.diag([0,0,1,1]).astype(complex)
rho=np.diag([1,1,-1,-1]).astype(complex)
for t in (.4,1.7):
    H=t*P-2.3*Q
    ck(f'H0 different spectrum and same symmetry t={t}',np.linalg.norm(H@rho-rho@H)<1e-14)

# A nontrivial hermitian chirality and gapped Wilson kernel.
gamma=np.diag([1.,1.,-1.,-1.]);rng=np.random.default_rng(20261010)
for rep in range(12):
    x=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    q,_=np.linalg.qr(x)
    eig=np.diag([1.,1.,-1.,-1.])
    eps=q@eig@q.conj().T
    a=.15
    D=(np.eye(4)+gamma@eps)/a
    gh=gamma@(np.eye(4)-a*D)
    ck(f'GW overlap {rep}',np.linalg.norm(gamma@D+D@gamma-a*D@gamma@D)<1e-11)
    ck(f'gamma5 hermiticity {rep}',np.linalg.norm(D.conj().T-gamma@D@gamma)<1e-11)
    ck(f'projector involution {rep}',np.linalg.norm(gh@gh-np.eye(4))<1e-11)
    ck(f'index trace convention {rep}',abs(.5*np.trace(gamma+gh)+.5*np.trace(eps))<1e-11)

# Free Wilson mass sign distinguishes origin from fifteen doubler corners.
for bits in range(16):
    n=bits.bit_count()
    r=1.;m0=1.
    M=2*r*n-m0
    ck(f'Wilson corner bits={bits}',(M<0)==(n==0))

# Gauge-invariant radial quartic stationary and curvature formula.
for a2,a4 in [(-1.,1.),(-3.,.5),(-.07,2.4)]:
    x0=-a2/(2*a4)
    ck(f'Higgs stationary ({a2},{a4})',abs(a2+2*a4*x0)<1e-13 and 2*a4>0)
    v=np.sqrt(2*x0)
    ck(f'Higgs radial curvature ({a2},{a4})',abs((a2+6*a4*v*v/2)-(-2*a2))<1e-12)

# On a cycle, same bulk hopping/topology but different boundary twists.
for N in (5,7,10):
    A=np.zeros((N,N),complex)
    A[np.arange(N-1),np.arange(1,N)]=1
    A[np.arange(1,N),np.arange(N-1)]=1
    plus=A.copy();plus[0,N-1]=plus[N-1,0]=1
    minus=A.copy();minus[0,N-1]=minus[N-1,0]=-1
    ep=np.sort(np.linalg.eigvalsh(plus));em=np.sort(np.linalg.eigvalsh(minus))
    refp=np.sort(2*np.cos(2*np.pi*np.arange(N)/N))
    refm=np.sort(2*np.cos((2*np.arange(N)+1)*np.pi/N))
    ck(f'cycle periodic spectrum N={N}',np.max(abs(ep-refp))<2e-12)
    ck(f'cycle antiperiodic spectrum N={N}',np.max(abs(em-refm))<2e-12)
    ck(f'boundary condition spectra differ N={N}',np.max(abs(ep-em))>1e-2)

# Simplicial complex triangle: d1*d2=0.
d1=np.array([[-1,0,1],[1,-1,0],[0,1,-1]],int)
d2=np.array([[1],[1],[1]],int)
ck('cellular boundary squared zero',np.array_equal(d1@d2,np.zeros((3,1),int)))

out={'classification':'FINITE_AXIOMATIC_DESCENT_REGRESSION',
    'checks_passed':len(checks),'checks_failed':0,'checks':checks,
    'scope':'Finite gauge-kernel, commutant, overlap identities, free Wilson corners, scalar stationary equations, and boundary spectra. Not a validation of interacting QICT continuum or physical coefficients.'}
root=Path(__file__).resolve().parents[1]
(root/'results'/'AXIOMATIC_DESCENT_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n')
print('axiomatic descent:',len(checks),'finite checks passed')
