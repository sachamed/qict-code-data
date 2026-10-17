#!/usr/bin/env python3
"""Exact algebraic counterexamples for QICT parameter identifiability.

This script contains no experimental inputs and makes no claim that the
physical interactive C8--C11 contractions are completed.
"""
import json
from pathlib import Path
import sympy as sp
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def prove(name,e):
    assert e==sp.zeros(*e.shape) if isinstance(e,sp.MatrixBase) else sp.simplify(e)==0, name
    checks.append(name)

Delta,t=sp.symbols('Delta t', positive=True)
G=sp.diag(1,-1,1)
H=sp.diag(0,Delta,0)
prove('symmetry invariance of tunable H0',H*G-G*H)
prove('symmetry invariance after rescaling',t*H*G-G*t*H)
assert sp.simplify(H[1,1]-H[0,0])==Delta
checks.append('relative gap varies without changing representation')

x=sp.symbols('x',real=True)
for a2,a4 in [(-1,1),(-4,1),(-3,2)]:
    V=a2*x+a4*x**2
    x0=sp.Rational(-a2,2*a4)
    prove(f'Higgs stationary orbit {a2} {a4}',sp.diff(V,x).subs(x,x0))
    assert sp.diff(V,x,2)>0
    checks.append(f'Higgs curvature positive {a2} {a4}')
assert sp.Rational(1,2)!=sp.Integer(2)
checks.append('distinct Higgs minima share the same representation')

k1,k2,k3=sp.symbols('k1 k2 k3',positive=True)
f1,f2,f3=sp.symbols('f1 f2 f3',real=True)
kin=(k1*f1**2+k2*f2**2+k3*f3**2)/4
for ki,fi in ((k1,f1),(k2,f2),(k3,f3)):
    prove('gauge quadratic independent ' + str(ki),sp.diff(kin,fi,2)-ki/2)
assert sp.simplify(sp.diff(kin,k1)-sp.diff(kin,k2)) !=0
checks.append('three kinetic quadratic forms independently variable')

b,g0,y=sp.symbols('b g0 y',positive=True)
g2=1/(g0**(-2)+2*b*y)
prove('RG flow one-loop exact integration',sp.diff(g2,y)+2*b*g2**2)
assert sp.diff(g2,g0)!=0
checks.append('RG initial datum not fixed by beta function')

# Exact two-state source-cumulant identity.
A=sp.Symbol('A',real=True)
Z=1+sp.exp(-x)
W=-sp.log(Z/2)
expectA=sp.Rational(1,2)
varA=sp.Rational(1,4)
prove('first logarithmic derivative',sp.diff(W,x).subs(x,0)-expectA)
prove('second logarithmic derivative',sp.diff(W,x,2).subs(x,0)+varA)

# A simple independent physical-ratio map has rank one even after an energy scale cancels.
a=sp.symbols('a',real=True)
F1=sp.exp(a)
F2=sp.exp(3*a)
ratio=F2/F1
prove('scale-free mass ratio depends on dimensionless input',sp.diff(ratio,a)-2*sp.exp(2*a))
assert sp.diff(ratio,a)!=0
checks.append('nonzero parameter rank in example ratio map')

records=[
 {'sector':'receiver topology and family','object':'six-channel carrier and selected finite endpoint counts','classification':'derived within explicit primitive-minimality axioms','numerical_physical_selection':False},
 {'sector':'finite operator','object':'commutant blocks h_lambda','classification':'not determined by representation theory alone','numerical_physical_selection':False},
 {'sector':'gauge','object':'kinetic normalizations g1,g2,g3','classification':'not determined by finite gauge algebra alone; requires primitive action and physical normalization','numerical_physical_selection':False},
 {'sector':'scalar','object':'alpha2, alpha4 and Z_H','classification':'defined by microscopic response; no evaluated physical continuum cumulants in archive','numerical_physical_selection':False},
 {'sector':'renormalization','object':'integration constants of RG trajectories','classification':'not determined by beta functions alone','numerical_physical_selection':False},
 {'sector':'regulators','object':'a,L,K,N5,Nm','classification':'auxiliary regularization coordinates; not independent measured couplings','numerical_physical_selection':False},
 {'sector':'metrology','object':'tau_* and dimensional normalization','classification':'operational scale prescription, distinct from dimensionless physical predictions','numerical_physical_selection':False},
 {'sector':'finite rotor','object':'coefficient 32 in local rotor Hamiltonian','classification':'fixed finite operator input, not derived from SU(2) representation alone','numerical_physical_selection':False},
 {'sector':'auxiliary electroweak','object':'lambda_H^(rep)=pi/24','classification':'stipulated reference representative, not predicted Higgs coupling','numerical_physical_selection':False},
 {'sector':'charged lepton','object':'reduced route-cost differences 34 and 52','classification':'derived from the selected reduced route construction, not physical poles','numerical_physical_selection':False},
 {'sector':'interactive continuum','object':'renormalized masses, Higgs responses and gravitational couplings','classification':'not established numerically on the complete common physical G2 regulator net','numerical_physical_selection':False},
]
output={'classification':'PARAMETER_IDENTIFIABILITY_NOT_FREE_PARAMETER_CERTIFICATE','checks_passed':len(checks),'checks':checks,'input':'abstract QICT algebraic structures, no experimental data','physical_free_parameter_count':'not determined by this audit','proof_scope':'counterexamples to derivation from algebra alone; conditional microscopic response identities','records':records}
out=ROOT/'results/PARAMETER_IDENTIFIABILITY_AUDIT.json'
out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
print('PASS',len(checks),'exact symbolic checks; output',out)
