#!/usr/bin/env python3
"""Small exact and high-precision checks for QICT family and spectral statements.

Does not assert interacting-continuum matching or generate a physical chi-square.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from mpmath import mp
mp.dps=65

# Irreducible S3 permutation module: 1 + 2, not a scalar commutant.
J=np.ones((3,3)); I=np.eye(3); P=J/3; Q=I-P
swap=np.array([[0,1,0],[1,0,0],[0,0,1]],dtype=float)
cycle=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=float)
A=2*I+0.7*J
assert np.allclose(A@swap,swap@A) and np.allclose(A@cycle,cycle@A)
eigs=np.linalg.eigvalsh(A)
assert np.allclose(eigs,[2,2,4.1])
omega=np.exp(2j*np.pi/3)
X=cycle.astype(complex);Z=np.diag([1,omega,omega**2]);
assert np.linalg.norm(X@Z-Z@X)>0.1
assert np.linalg.norm(A@Z-Z@A)>0.1
assert np.allclose(P@P,P) and np.allclose(Q@Q,Q)

# Exact dimensions of the commutants from nine linear matrix-entry equations.
import sympy as sp
S=sp.Matrix([[0,1,0],[1,0,0],[0,0,1]])
C=sp.Matrix([[0,0,1],[1,0,0],[0,1,0]])
w=(-1+sp.sqrt(-3))/2
W=sp.diag(1,w,w**2)
syms=sp.symbols('x0:9'); Y=sp.Matrix(3,3,syms)
def commutant_dim(gens):
    constraints=[]
    for gen in gens: constraints.extend(list(Y*gen-gen*Y))
    coeffs,_=sp.linear_eq_to_matrix(constraints,syms)
    return len(syms)-coeffs.rank()
assert commutant_dim([C,S])==2
assert commutant_dim([C,W])==1

# Deterministic unitary matrices: exact six-column receiver, moment/resolvent tail.
N=12; v=np.zeros((N,6),complex);v[:6,:]=np.eye(6)
theta=np.array([0.04,0.2,0.51,1.0,1.61,2.3,2.8,3.4,4.1,4.7,5.2,5.9])
u=np.diag(np.exp(1j*theta)); du=0.001
ut=np.diag(np.exp(1j*(theta+du)))
v2=v.copy() # same receiver, isolate unitary error
z=1.2+0j; r=abs(z); depth=88
mom=[v.conj().T @ np.linalg.matrix_power(u,n) @ v for n in range(depth+1)]
g=v.conj().T@np.linalg.inv(z*np.eye(N)-u)@v
g2=v2.conj().T@np.linalg.inv(z*np.eye(N)-ut)@v2
gdepth=sum(c/z**(n+1) for n,c in enumerate(mom))
tail=np.linalg.norm(g-gdepth,ord=2)
tailbound=r**(-depth-1)/(r-1)
delta=np.linalg.norm(u-ut,ord=2)
delta_res=np.linalg.norm(g-g2,ord=2)
assert tail<=tailbound*(1+1e-7) + 1e-12
assert delta_res<=delta/(r-1)**2*(1+1e-9)
# Positive compressed Toeplitz matrices from these moments
T=np.block([[mom[i-j] if i>=j else mom[j-i].conj().T for j in range(5)] for i in range(5)])
assert np.linalg.eigvalsh(T).min()>-1e-10
D=2*np.eye(6)-mom[1]-mom[1].conj().T
for n in range(1,depth+1):
    Dn=2*np.eye(6)-mom[n]-mom[n].conj().T
    assert np.linalg.eigvalsh(Dn).min()>-1e-8
    assert np.linalg.eigvalsh(n*n*D-Dn).min()>-1e-8

# Arbitrary precision alpha and reduced mass ratios; numbers not experimentally fitted by this code.
Rbcc=mp.gamma(mp.mpf('0.25'))**4/(32*mp.pi**3)
rotor_certificate=json.loads((Path(__file__).resolve().parents[1]/'results'/'CERTIFIED_REDUCED_LEPTON_INTERVALS.json').read_text(encoding='utf8'))
Wlo,Whi=(mp.mpf(x) for x in rotor_certificate['rotor_wilson_enclosure'])
assert Wlo<Whi
W=(Wlo+Whi)/2  # midpoint for the numerical check; certified bounds are stored separately
fac=Rbcc*mp.sqrt(W)*mp.cos(mp.pi/63)
conn=fac-mp.log(1-fac**3)/96
rmu=mp.exp(34*conn);rtau=mp.exp(52*conn)
# Sensitivity to an exact interval on the *input* rather than a fabricated physical error.
eps=mp.mpf('1e-12')
F=lambda t: t-mp.log(1-t**3)/96
lo=F(fac-eps);hi=F(fac+eps)
assert lo<conn<hi
rmu_interval=[mp.exp(34*lo),mp.exp(34*hi)]
rtau_interval=[mp.exp(52*lo),mp.exp(52*hi)]
# These intervals are conditional demonstrations for the reduced coefficient only.

# Closed-form one-parameter neutrino projection on finite coordinate shape.
s=[mp.mpf(x) for x in ['0.1895171061','0.2674032426','0.4706252560']]
a1=s[1]**2-s[0]**2;a2=s[2]**2-s[0]**2
y1=mp.mpf('7.49e-5');y2=mp.mpf('2.534e-3')
s1=mp.mpf('0.19e-5');s2=mp.mpf('0.024e-3')
x=(a1*y1/s1**2+a2*y2/s2**2)/(a1**2/s1**2+a2**2/s2**2)
chi_direct=(x*a1-y1)**2/s1**2+(x*a2-y2)**2/s2**2
chi_closed=(a2*y1-a1*y2)**2/(a2**2*s1**2+a1**2*s2**2)
assert abs(chi_direct-chi_closed)<mp.mpf('1e-38')
assert abs(chi_closed-mp.mpf('6814.89274'))<mp.mpf('1e-4')

out={
  'scope':'finite six-channel and reduced-sector checks, not interacting physical predictions',
  'S3_commutant':{'eigenvalues':eigs.tolist(),'commutes_S3':True,'not_full_Weyl':True,'exact_S3_commutant_dimension':2,'exact_Weyl_commutant_dimension':1},
  'resolvent':{'depth':depth,'radius':r,'tail_actual':float(tail),'tail_bound':float(tailbound),'difference_actual':float(delta_res),'difference_bound':float(delta/(r-1)**2)},
  'reduced_family':{'alpha_fac':mp.nstr(fac,28),'alpha_conn':mp.nstr(conn,28),
    'mu_e':mp.nstr(rmu,25),'tau_e':mp.nstr(rtau,25),
    'illustrative_alpha_input_halfwidth':'1e-12',
    'mu_e_alpha_interval':[mp.nstr(t,25) for t in rmu_interval],
    'tau_e_alpha_interval':[mp.nstr(t,25) for t in rtau_interval],
    'interval_status':'illustrative mathematical propagation; NOT a model-certified physical uncertainty'},
  'neutral_finite':{'shape_ratio':mp.nstr(a1/a2,20),'best_scale_eV':mp.nstr(mp.sqrt(x),17),
    'chi2_scale_profile':mp.nstr(chi_closed,21),'statistics_scope':'NuFIT-6.0 two-gap diagonal Gaussian projection of finite coordinates; not interacting continuum fit'}
}
target=Path(__file__).resolve().parents[1]/'results'/'FAMILY_RESOLVENT_FLAVOR_LIGHT_CERTIFICATE.json'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
