#!/usr/bin/env python3
"""Deterministic, small-matrix checks for supplementary analytic bounds.

This is a structural test of the statements, not an interacting QICT G2 run
and not a source of measured masses or Standard Model predictions.
"""
import json
import math
from pathlib import Path
import numpy as np

SX = np.array([[0., 1.], [1., 0.]], dtype=complex)
SZ = np.diag([1., -1.]).astype(complex)
I2 = np.eye(2, dtype=complex)

def e_pauli(angle, matrix):
    return np.cos(angle)*I2 - 1j*np.sin(angle)*matrix

def exact_dirac(t,m,p):
    e=np.hypot(m,p)
    if e==0: return I2
    return np.cos(t*e)*I2 -1j*np.sin(t*e)*(m*SZ+p*SX)/e

# The theorem's error bound includes the fractional final time step.
check=[]
for a in [0.12,0.06,0.03,0.015]:
    m=0.3;P=1.1;T=1.3
    maxerr=0.0
    for t in np.linspace(0,T,17):
        n=int(np.floor(t/a))
        for p in np.linspace(-P,P,51):
            Ua=e_pauli(a*m,SZ)@e_pauli(np.sin(a*p),SX)
            maxerr=max(maxerr,float(np.linalg.norm(np.linalg.matrix_power(Ua,n)-exact_dirac(t,m,p),2)))
    bound=T*(a*abs(m)*P+a*a*P**3/6)+a*(abs(m)+P)
    if maxerr>bound+1e-12:
        raise AssertionError(('Dirac bound violated',a,maxerr,bound))
    check.append({'a':a,'sup_sampled_error':maxerr,'rigorous_bound':bound})

M=4
phases=np.linspace(-0.9,0.8,2*M+2)
rows=[np.ones(len(phases))]
for k in range(1,M+1):
    rows.extend([np.cos(k*phases),np.sin(k*phases)])
A=np.vstack(rows)
_,_,vh=np.linalg.svd(A,full_matrices=True)
w=vh[-1]
assert np.min(np.abs(w))>1e-8
pos=np.maximum(w,0);neg=np.maximum(-w,0)
pos/=pos.sum();neg/=neg.sum()
cplus=np.array([np.sum(pos*np.exp(1j*k*phases)) for k in range(M+1)])
cminus=np.array([np.sum(neg*np.exp(1j*k*phases)) for k in range(M+1)])
res=float(np.max(np.abs(cplus-cminus)))
assert res<1e-8
aplus=float(np.min(phases[pos>1e-12]));aminus=float(np.min(phases[neg>1e-12]))
assert abs(aplus-aminus)>1e-4

hi=206.7;hj=1.; di=0.1;dj=0.001
lo=(hi-di)/(hj+dj);up=(hi+di)/(hj-dj)
for x in np.linspace(hi-di,hi+di,7):
    for y in np.linspace(hj-dj,hj+dj,7):
        assert lo-1e-12 <=x/y <=up+1e-12

result={'description':'analytical small-matrix benchmarks; not particle-mass predictions',
        'split_step':check,
        'finite_moment_ambiguity':{'moment_depth':M,'max_fourier_residual':res,
            'first_edge':aplus,'second_edge':aminus,'different_edges':True},
        'illustrative_positive_ratio_interval':{'lower':lo,'upper':up}}
folder=Path(__file__).resolve().parent.parent/'results'
folder.mkdir(parents=True,exist_ok=True)
(folder/'analytical_finite_depth_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
