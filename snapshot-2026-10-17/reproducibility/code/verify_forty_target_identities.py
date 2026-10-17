#!/usr/bin/env python3
"""Exact algebraic identities and nonuniqueness witnesses for QICT S123.
Not a simulation of the missing interacting microscopic many-body theory.
"""
import json
from pathlib import Path
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
Q=sp.Rational
p_e,p_mu,p_tau=55,21,3
assert 26*(p_e-p_mu)==17*(p_e-p_tau)==884
YQ,Yu,Yd,YL,Ye,Yn=(Q(1,6),Q(2,3),-Q(1,3),-Q(1,2),-1,0)
a={
    'SU2_SU2_U1':3*YQ+YL,
    'SU3_SU3_U1':2*YQ-Yu-Yd,
    'gravity_U1':6*YQ-3*Yu-3*Yd+2*YL-Ye-Yn,
    'U1_cubed':6*YQ**3-3*Yu**3-3*Yd**3+2*YL**3-Ye**3-Yn**3,
}
assert all(v==0 for v in a.values()),a
assert (3+1)%2==0

# Fix spectra; change relative *left* singular frame to vary mixing.
s=np.array([0.04, 0.38, 2.1]); t=np.array([0.016,0.11,1.63])
D1=np.diag(s); D2=np.diag(t)
th12,th13,th23=.23,.17,.39

def CKM(delta):
 c12,s12=np.cos(th12),np.sin(th12)
 c13,s13=np.cos(th13),np.sin(th13)
 c23,s23=np.cos(th23),np.sin(th23)
 return np.array([
 [c12*c13,s12*c13,s13*np.exp(-1j*delta)],
 [-s12*c23-c12*s23*s13*np.exp(1j*delta),c12*c23-s12*s23*s13*np.exp(1j*delta),s23*c13],
 [s12*s23-c12*c23*s13*np.exp(1j*delta),-c12*s23-s12*c23*s13*np.exp(1j*delta),c23*c13]
 ],dtype=complex)

V0,V1=CKM(0.),CKM(.7)
assert np.linalg.norm(V1.conj().T@V1-np.eye(3))<1e-14
assert np.allclose(np.linalg.svd(V0@D2,compute_uv=False),np.linalg.svd(V1@D2,compute_uv=False),rtol=1e-13)
assert np.linalg.norm(V1-V0)>1e-2
j=lambda W: (W[0,0]*W[1,1]*np.conj(W[0,1])*np.conj(W[1,0])).imag
assert abs(j(V0))<1e-14 and abs(j(V1))>1e-4

# Exact Abelian one-loop integration identity.
x,x0,b,L=sp.symbols('x x0 b L', nonzero=True)
y=1/x0-b*L/(8*sp.pi**2)
assert sp.simplify(sp.diff(y,L)+b/(8*sp.pi**2))==0
# Higgs derivative and arbitrary quartic coefficients.
mu2,lam,z=sp.symbols('mu2 lam z', real=True)
P=mu2*z+lam*z*z
minimum=-mu2/(2*lam)
assert sp.simplify(sp.diff(P,z).subs(z,minimum))==0
assert sp.diff(P,z,2)==2*lam

alpha=.15680646632668089
rm=float(np.exp(34*alpha));rt=float(np.exp(52*alpha))
# logarithmic form avoids avoidable large integer power numerical instability
assert abs(26*np.log(rm)-17*np.log(rt))<1e-12

ratio_mu_ref=206.768282708
ratio_tau_ref=3477.365262
r_mu=ratio_mu_ref/rm;r_tau=ratio_tau_ref/rt
min_b_mu=(r_mu-1)/(r_mu+1);min_b_tau=(r_tau-1)/(r_tau+1)
assert min_b_mu>0 and min_b_tau>0
assert abs(min_b_mu-8.9452e-5)<1e-8
assert abs(min_b_tau-4.6964e-5)<1e-8

out={
 'classification':'EXACT_IDENTITIES_AND_FINITE_NONUNIQUENESS_WITNESSES',
 'tests':14,
 'anomaly_coefficients':{k:str(v) for k,v in a.items()},
 'SU2_doublets_per_family':4,
 'reduced_mass_sum_rule_exponents':[26,17,884],
 'alpha_conn_reference':alpha,
 'R_mu_e_reduced':rm,'R_tau_e_reduced':rt,
 'J_for_delta_0':float(j(V0)),'J_for_delta_0p7':float(j(V1)),
 'beta_empirical_min_mu_e':float(min_b_mu),'beta_empirical_min_tau_e':float(min_b_tau),
 'scope':'Algebraic results for stipulated finite represented matter content, constructed unitary mixing family and specified reduced route costs; empirical minima are comparison constraints, not predicted self-energies.',
 'missing':'No nonperturbative QICT Yukawa, pole-mass, chiral-measure, continuum or graviton computation performed.'
}
p=ROOT/'results/FORTY_TARGET_IDENTITIES.json';p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print('QICT S123 checks PASSED; anomaly sums zero; family mixing nonunique; reduced sum rule exact')
print('empirical lower bounds',min_b_mu,min_b_tau)
