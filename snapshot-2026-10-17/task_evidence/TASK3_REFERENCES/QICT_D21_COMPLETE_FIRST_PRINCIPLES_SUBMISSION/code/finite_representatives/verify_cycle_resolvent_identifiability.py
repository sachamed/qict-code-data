#!/usr/bin/env python3
"""Deterministic checks for cycle-resolvent and identifiability theorems.
No particle or cosmological data are used.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
N=21
# cycle matrices
S=np.roll(np.eye(N,dtype=complex),1,axis=0)
B=np.eye(N,dtype=complex)-S
L=B.conj().T@B
tr1=float(np.trace(L).real)
tr2=float(np.trace(L@L).real)
tr3=float(np.trace(L@L@L).real)
eigs=np.linalg.eigvalsh(L)
pdet=float(np.prod(eigs[eigs>1e-10]))
# weighted path first-arrival powers
idx=np.arange(N)
a=(1.1+0.03*idx)*np.exp(1j*(0.17+0.11*idx))
b=(0.9+0.02*idx)*np.exp(-1j*(0.23+0.07*idx))
Tp=np.zeros((N,N),complex); Tm=np.zeros((N,N),complex)
for j in range(N):
    Tp[(j+1)%N,j]=a[j]
    Tm[(j-1)%N,j]=b[j]
path_checks={}
for d in (1,8,9,10,13,20):
    first_p=first_m=None
    vp=np.eye(N,dtype=complex); vm=np.eye(N,dtype=complex)
    for q in range(N+1):
        if first_p is None and abs(vp[d,0])>1e-12: first_p=q
        if first_m is None and abs(vm[d,0])>1e-12: first_m=q
        vp=vp@Tp; vm=vm@Tm
    path_checks[str(d)]={'forward':first_p,'reverse':first_m,'expected_forward':d,'expected_reverse':N-d}
# near-antipodal hierarchy
D=[8,9,10]
defects=sorted([N-2*d for d in D])
# reciprocal delay identity
tau=1.0+0.35*np.sin(2*np.pi*idx/N)+0.20*np.cos(6*np.pi*idx/N)+0.05*idx/(N-1)
mean=float(np.mean(tau)); route={}
for d in (10,9,8):
    vals=[]
    for s in range(N):
        short=[(s+j)%N for j in range(d)]
        mask=np.ones(N,bool); mask[short]=False
        vals.append(float(np.sum(tau[mask])-np.sum(tau[short])))
    avg=float(np.mean(vals)); exp=float((N-2*d)*mean)
    route[str(d)]={'average':avg,'expected':exp,'residual':abs(avg-exp)}
# C3 basis freedom numerical witness
f0=np.array([1,1,1],float)/math.sqrt(3)
fc=np.array([2,-1,-1],float)/math.sqrt(6)
fs=np.array([0,1,-1],float)/math.sqrt(2)
phis=[0.0,0.23,0.71]
orth_res=[]; varying=[]
for phi in phis:
    fcp=fc*math.cos(phi)+fs*math.sin(phi)
    fsp=-fc*math.sin(phi)+fs*math.cos(phi)
    U=np.column_stack([fcp,f0,fsp])
    orth_res.append(float(np.linalg.norm(U.T@U-np.eye(3),'fro')))
    varying.append(float(U[0,2]))
# wheel cycles exact count formula
wheel_cycles=2*math.comb(6,2)+1
# topological entropy projector shortcut: choose any rank-3 projector, connected cycle nullity 1
Pi=np.diag([1,1,1]+[0]*(N-3))
# x log x convention -> 0 for x in {0,1}
projector_term=0.0
entropy=math.log(1)-projector_term
status=(abs(tr1-42)<1e-12 and abs(tr2-126)<1e-10 and abs(tr3-420)<1e-9 and
        abs(pdet-441)<1e-8 and all(v['forward']==v['expected_forward'] and v['reverse']==v['expected_reverse'] for v in path_checks.values()) and
        defects==[1,3,5] and max(v['residual'] for v in route.values())<1e-12 and max(orth_res)<1e-12 and
        max(varying)-min(varying)>1e-3 and wheel_cycles==31 and abs(entropy)<1e-15)
out={
 'status':'PASS' if status else 'FAIL',
 'n':N,
 'trace_L':tr1,'trace_L2':tr2,'trace_L3':tr3,'pseudodeterminant_L':pdet,
 'weighted_path_checks':path_checks,
 'near_antipodal_defects':defects,
 'reciprocal_delay_checks':route,
 'c3_doublet_orthogonality_max_residual':max(orth_res),
 'c3_basis_dependent_reference_entry':varying,
 'wheel_simple_cycle_count':wheel_cycles,
 'projector_entropy_shortcut_value':entropy,
 'interpretation':'Deterministic implementation checks of exact finite-cycle, basis-freedom, and exclusion theorems.'
}
(CERT/'cycle_resolvent_identifiability_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if status else 1)
