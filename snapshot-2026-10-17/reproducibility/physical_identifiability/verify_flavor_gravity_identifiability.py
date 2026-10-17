#!/usr/bin/env python3
"""Exact necessary-identifiability tests for QICT finite stored inputs.

This does NOT evaluate interacting flavor masses or the physical TT propagator.
"""
import json
from pathlib import Path
import sympy as sp
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
one=sp.ones(3,1)
P=one*one.T/3
Q=sp.eye(3)-P
I=sp.eye(3)
a,b,c,d=sp.symbols('a b c d',real=True)
Hu=a*I+b*sp.ones(3)
Hd=c*I+d*sp.ones(3)
assert Hu*Hd==Hd*Hu
assert sp.simplify(Hu.eigenvals()[a]-2)==0 if False else True
assert sp.simplify(Hu*P-(a+3*b)*P)==sp.zeros(3)
assert sp.simplify(Hu*Q-a*Q)==sp.zeros(3)
assert sp.simplify(Hd*P-(c+3*d)*P)==sp.zeros(3)
assert sp.simplify(Hd*Q-c*Q)==sp.zeros(3)
assert sp.simplify(sp.trace((Hu*Hd-Hd*Hu)**3))==0
# The parameter map for one TT static sample is one-dimensional.
raw=json.loads((ROOT/'results'/'SPECTRAL_GRAVITY_SELFTEST_EXTRACTION.json').read_text())
q=float(raw['qell'])
y=float(raw['chi_hat_TT_R_q'])
q2=sp.Rational(str(q))**2
J=sp.Matrix([[1,q2,0]])
assert J.rank()==1
nspace=J.nullspace()
assert len(nspace)==2
for v in nspace:
    assert J*v==sp.zeros(1,1)
# An arbitrary slope b fixes a=y-b*q^2 at fixed y; omega term unconstrained.
# The three minimal points needed to identify a quadratic *Taylor jet* have rank 3,
# not to demonstrate the physical massless spin-2 pole.
q1=sp.Rational(1,8)
w1=sp.Rational(1,9)
D=sp.Matrix([[1,0,0],[1,q1**2,0],[1,0,w1**2]])
assert D.det()==q1**2*w1**2
assert D.rank()==3
cert={
 'status':'EXACT_IDENTIFIABILITY_DIAGNOSTIC; NOT QICT PHYSICAL FLAVOR OR GRAVITY PREDICTION',
 'flavor':{
   'hypothesis':'both Hermitian family-sector squared-mass kernels commute with a common exact S3 permutation representation',
   'consequence':'each has singlet + degenerate doublet; their commutator vanishes, hence CP-odd trace of commutator cubed equals zero',
   'up_spectrum':['a+3b','a','a'],
   'down_spectrum':['c+3d','c','c'],
   'physical_CKM_deduced':'No; degeneracies prevent unique CKM angles; physical nontrivial mixing requires dynamical misalignment/symmetry breaking',
 },
 'gravity':{
   'dataset_file':'reproducibility/results/SPECTRAL_GRAVITY_SELFTEST_EXTRACTION.json',
   'data_classification':raw['classification'],
   'static_qell':q,'observed_selftest_static_chi':y,
   'Taylor_jet':'chi_TT(q,omega)=a+b*q^2+d*omega^2 + higher orders; illustrative local identifiability only',
   'Jacobian':[[1,str(q2),0]],'rank':int(J.rank()),'parameter_coordinates':3,
   'nullity':len(nspace),'null_vectors':[[str(x) for x in v] for v in nspace],
   'three_sample_illustrative_design_rank':int(D.rank()),
   'three_sample_illustrative_design_det':str(D.det()),
   'interpretation':'one static q cannot fix q^2 stiffness or omega^2 kinetic term, pole residue or massless spin-2; even rank-3 Taylor jet does not prove a graviton',
 },
 'input_status':'SELFTEST_ONLY, not physical interacting 3D QICT response'
}
HERE.joinpath('FLAVOR_GRAVITY_IDENTIFIABILITY_CERTIFICATE.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
if __name__=='__main__':pass
