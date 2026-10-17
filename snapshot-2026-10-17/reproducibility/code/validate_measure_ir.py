#!/usr/bin/env python3
"""Finite-matrix checks of chiral determinant-line estimates and tensor IR bounds.

These are reproducible mathematical checks with reference matrices, not physical QICT G2 predictions.
"""
from pathlib import Path
import json
import numpy as np
from numpy.linalg import eigh, norm

ROOT = Path(__file__).resolve().parents[1]
rng=np.random.default_rng(20261010)
checks=0

def assert_check(cond,msg):
    global checks
    if not bool(cond): raise AssertionError(msg)
    checks+=1

def projector(H):
    w,V=eigh(H)
    return (V[:,w>0]@V[:,w>0].conj().T)

sx=np.array([[0,1],[1,0]],complex)
sy=np.array([[0,-1j],[1j,0]],complex)
sz=np.diag([1.,-1.]).astype(complex)

def spherical(theta,phi):
    return np.sin(theta)*np.cos(phi)*sx + np.sin(theta)*np.sin(phi)*sy + np.cos(theta)*sz

for theta in (0.3,.8,1.6,2.5):
    for phi in (0.2,1.2,2.3):
        H=spherical(theta,phi)
        P=projector(H)
        assert_check(norm(P@P-P)<1e-13,'projector')
        h=1e-5
        Pt=(projector(spherical(theta+h,phi))-projector(spherical(theta-h,phi)))/(2*h)
        Pp=(projector(spherical(theta,phi+h))-projector(spherical(theta,phi-h)))/(2*h)
        Ht=(spherical(theta+h,phi)-spherical(theta-h,phi))/(2*h)
        Hp=(spherical(theta,phi+h)-spherical(theta,phi-h))/(2*h)
        curv=(1j*np.trace(P@(Pt@Pp-Pp@Pt))).real
        assert_check(abs(curv+.5*np.sin(theta))<2e-9,'Chern curvature')
        assert_check(norm(Pt,2) <= norm(Ht,2)/2+1e-9,'theta differential bound')
        assert_check(norm(Pp,2) <= norm(Hp,2)/2+1e-9,'phi differential bound')
        assert_check(abs(curv) <= .5*norm(Ht,2)*norm(Hp,2)+1e-8,'Berry bound')
# General random gapped matrices and independent numerical derivatives
for d in (4,6,8):
    Z=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));Q,_=np.linalg.qr(Z)
    lam=np.r_[np.linspace(-2.,-1.,d//2),np.linspace(1.,2.,d-d//2)]
    H=Q@np.diag(lam)@Q.conj().T
    g=np.min(abs(lam))
    A=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));A=(A+A.conj().T)/2
    h=1.e-5
    DP=(projector(H+h*A)-projector(H-h*A))/(2*h)
    assert_check(norm(DP,2)<=norm(A,2)/(2*g)+1e-8,'random gap derivative')

for alpha in (.25,.5,1.,2.):
    Z=0.6;M=0.9;C=20.0;Lambda0=1.0
    # spectral measure with finite atoms satisfying the assumed low-mass bound
    atom_pos=np.array([.2,.4,2.0]); atom_mass=np.array([.2,.3,.4])
    assert_check(all(sum(m for x,m in zip(atom_pos,atom_mass) if x<cut)<=C*cut**alpha + 1e-12 for cut in np.linspace(.01,1,80)), 'IR mass envelope')
    for s in (1e-5,1e-3,1e-2):
        chi=Z/s+np.sum(atom_mass/(s+atom_pos))
        lam=min(Lambda0,(M*s/(alpha*C))**(1/(alpha+1)))
        err=s*chi-Z
        bd=C*lam**alpha+M*s/lam
        assert_check(err>=0 and err<=bd+1e-12,'IR spectral rate')
        if lam<Lambda0:
            opt=(1+alpha)*C*(M*s/(alpha*C))**(alpha/(alpha+1))
            assert_check(err<=opt+1e-12,'optimized IR bound')

for ma in (1.,.1,.01,1e-4):
    s=1e-3
    assert_check(abs(s/(s+ma*ma)-(1-(ma*ma)/(s+ma*ma)))<1e-12,'order-of-limits witness')

out={'classification':'MATHEMATICAL_FINITE_MATRIX_REGRESSION_NOT_PHYSICAL_G2',
    'checks_passed':checks,'failures':0,'random_seed':20261010,
    'scope':'Finite gap projector derivative/curvature, Chern example, TT spectral mass envelope, order-of-limits witness. No numerical QICT physical mass/coupling prediction.'}
path=ROOT/'results/MEASURE_IR_VALIDATION.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(f'{checks} chiral-measure and infrared tensor checks passed')
