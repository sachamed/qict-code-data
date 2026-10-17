#!/usr/bin/env python3
"""Numerical checks of spectral arc bounds, Schur decay and nested sampling."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import binomtest, chi2

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'reproducibility/results/spectral_memory_validation'
RESULT.mkdir(parents=True,exist_ok=True)

# Hand-checkable two-level unitary witness.
v=np.array([np.sqrt(3)/2, 0.5])
W=np.diag([1.,-1.])
x=float(v @ W @ v)
p_arc=float(v[0]**2)
assert np.isclose(x,0.5,atol=1e-14)
assert np.isclose(p_arc,0.75,atol=1e-14)
assert x-1e-14 <= p_arc <= 1.

# A non-diagonal finite unitary and several whitened source directions.
rng=np.random.Generator(np.random.PCG64(20261023))
Z=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
Q,R=np.linalg.qr(Z); U=Q@np.diag(np.diag(R)/np.abs(np.diag(R)))
Y=rng.normal(size=(8,3))+1j*rng.normal(size=(8,3))
V,_=np.linalg.qr(Y)
lambda_,S=np.linalg.eig(U)  # normal; numerical basis approximately unitary
assert np.linalg.norm(S.conj().T@S-np.eye(8))<1e-11
C1=V.conj().T@U@V
arc_checks=[]
for delta in (np.pi/5,np.pi/3,np.pi/2,2*np.pi/3):
    indicator=(np.abs(np.angle(lambda_))<=delta).astype(float)
    E=S@np.diag(indicator)@S.conj().T
    for j in range(3):
        p_arc=float(np.real(np.vdot(V[:,j],E@V[:,j])))
        xx=float(np.real(C1[j,j]))
        lower=max(0.,(xx-np.cos(delta))/(1-np.cos(delta)))
        upper=min(1.,(xx+1)/(1+np.cos(delta)))
        assert lower-1e-11<=p_arc<=upper+1e-11,(delta,j,lower,p_arc,upper)
        arc_checks.append({'arc_half_angle':float(delta),'source':j,'certified_low':lower,'true_weight':p_arc,'certified_high':upper})

# Uniform geometric-power bound, including a unit-circle point.
Z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
Q,R=np.linalg.qr(Z); U=Q@np.diag(np.diag(R)/np.abs(np.diag(R)))
q=0.68; D=q*U
B=rng.normal(size=(2,6)); C=rng.normal(size=(6,2))
M=np.linalg.norm(B,2)*np.linalg.norm(C,2)
remainders=[]
for zz in (np.exp(1j*0.32), 1.07*np.exp(1j*1.2)):
    radius=abs(zz);K=35
    approx=sum((B@np.linalg.matrix_power(D,k)@C)*zz**(-k-1) for k in range(K))
    exact=B@np.linalg.solve(zz*np.eye(6)-D,C)
    diff=float(np.linalg.norm(approx-exact,2))
    upper=float(M*(q/radius)**K/(radius-q))
    assert diff <= upper+1e-11
    remainders.append({'radius':radius,'truncation_order':K,'observed_error':diff,'proved_upper':upper})

samples=json.loads((ROOT/'reproducibility/results/reduced_hierarchy_tests/REDUCED_HIERARCHY_TESTS.json').read_text())['Monte_Carlo']
nested=[]
for i in (0,2):
    narrow,broad=samples[i:i+2]
    assert narrow['seed']==broad['seed']
    assert narrow['T_threshold']<broad['T_threshold']
    N=narrow['samples']; a=narrow['hits']; b=broad['hits']
    assert N==broad['samples'] and 0<=a<=b<=N
    pn=narrow['analytic_probability']; pb=broad['analytic_probability']
    O=np.array([a,b-a,N-b],float)
    E=N*np.array([pn,pb-pn,1-pb])
    deviance=float(2*np.sum(np.where(O>0,O*np.log(O/E),0)))
    nested.append({'seed':narrow['seed'],'narrow_hits':a,'broad_hits':b,
                   'expected_narrow':E[0],'expected_broad':N*pb,
                   'multinomial_deviance':deviance,
                   'chi_square_2_asymptotic_p':float(chi2.sf(deviance,2)),
                   'exact_conditional_two_sided_p':float(binomtest(a,b,pn/pb).pvalue),
                   'exact_marginal_broad_two_sided_p':float(binomtest(b,N,pb).pvalue)})
assert abs(nested[0]['multinomial_deviance']-3.34348242796)<0.00001
assert abs(nested[1]['multinomial_deviance']-13.5248869154)<0.00001
result={'status':'PASS_FINITE_SPECTRAL_WEIGHT_MEMORY_AND_NESTED_STATISTICS',
        'finite_witness':{'C1':x,'arc_weight':p_arc},
        'arc_checks':arc_checks,'unit_circle_decay_tests':remainders,
        'nested_sampling':nested,
        'scope':'mathematical implications of stated finite-dimensional hypotheses, not full physical identification'}
# First example arc weight to avoid accidental variable shadowing.
result['finite_witness']['arc_weight']=0.75
(RESULT/'SPECTRAL_MEMORY_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'nested_sampling':nested,'spectral_arc_checks':len(arc_checks),'memory_checks':len(remainders)},indent=2))
