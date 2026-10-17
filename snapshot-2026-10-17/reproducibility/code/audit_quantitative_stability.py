#!/usr/bin/env python3
"""Independently test the quantitative norm bounds and rare-event diagnostics.

Finite-dimensional controls are exact up to floating-point rounding.  Their
validity does not imply physical regulator convergence or journal acceptance.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

R=Path(__file__).resolve().parents[2]
OUT=R/'reproducibility/results/quantitative_stability'
OUT.mkdir(parents=True,exist_ok=True)
seed=20261014
rng=np.random.Generator(np.random.PCG64(seed))

def orthonormal_from(F):
    G=F.conj().T@F
    eigs,U=np.linalg.eigh(G)
    return F@(U@np.diag(eigs**-0.5)@U.conj().T),G

# Isometries on one common Hilbert space, with a nontrivial source Gram matrix.
F=rng.normal(size=(8,3))+1j*rng.normal(size=(8,3))
Ft=F+0.04*(rng.normal(size=F.shape)+1j*rng.normal(size=F.shape))
V,G=orthonormal_from(F)
Vt,Gt=orthonormal_from(Ft)
g=float(np.linalg.eigvalsh(G).min())
d=float(np.linalg.norm(G-Gt,2))
assert d<g
v_error=float(np.linalg.norm(V-Vt,2))
v_upper=float(np.linalg.norm(F-Ft,2)/np.sqrt(g)+np.linalg.norm(Ft,2)*d/(2*(g-d)**1.5))
assert v_error <= v_upper + 2e-13

def unitary(A):
    q,r=np.linalg.qr(A)
    return q@np.diag(np.diag(r)/np.abs(np.diag(r)))
W=unitary(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
Wt=unitary(W+0.006*(rng.normal(size=W.shape)+1j*rng.normal(size=W.shape)))
w_error=float(np.linalg.norm(W-Wt,2))
checks=[]
for n in (-8,-4,-1,0,1,4,8):
    C=V.conj().T@np.linalg.matrix_power(W,n)@V
    Ct=Vt.conj().T@np.linalg.matrix_power(Wt,n)@Vt
    err=float(np.linalg.norm(C-Ct,2));upper=2*v_error+abs(n)*w_error
    assert err <=upper+1e-12,(n,err,upper)
    checks.append({'n':n,'error':err,'certified_upper':upper})
assert np.linalg.norm(V.conj().T@V-np.eye(3),2)<1e-12
# Block Toeplitz positivity for matrix-valued moments.
moments={k:V.conj().T@np.linalg.matrix_power(W,k)@V for k in range(-5,6)}
T=np.block([[moments[k-j] for k in range(6)] for j in range(6)])
eig_min=float(np.linalg.eigvalsh((T+T.conj().T)/2).min())
assert eig_min >-2e-12

# For a strict contraction, test analytic tail with conservative coefficient error.
D=0.78*unitary(rng.normal(size=(5,5))+1j*rng.normal(size=(5,5)))
B=rng.normal(size=(3,5));C=rng.normal(size=(5,3))
z=1.35+.31j;r=abs(z);K=13
mom=[B@np.linalg.matrix_power(D,k)@C for k in range(K)]
perturb=[1e-5*np.eye(3) for k in range(K)]
mhat=[a+b for a,b in zip(mom,perturb)]
sigma=B@np.linalg.inv(z*np.eye(5)-D)@C
sigmahat=sum(a*z**(-k-1) for k,a in enumerate(mhat))
actual=float(np.linalg.norm(sigma-sigmahat,2))
upper=float(sum(np.linalg.norm(e,2)*r**(-k-1) for k,e in enumerate(perturb))
            +np.linalg.norm(B,2)*np.linalg.norm(C,2)*r**(-K-1)/(1-r**-1))
assert actual <=upper+1e-12

# Audit of old counts and independently seeded new runs against exact analytic probabilities.
old=json.loads((R/'reproducibility/results/reduced_hierarchy_tests/REDUCED_HIERARCHY_TESTS.json').read_text())['Monte_Carlo']
old_summary=[];new_summary=[]
for row in old:
    a,b=row['acceptance_alpha_interval'];p=(b-a)/row['upper'];N=row['samples'];k=row['hits']
    score=(k-N*p)/np.sqrt(N*p*(1-p));pval=binomtest(k,N,p).pvalue
    old_summary.append({'seed':row['seed'],'threshold':row['threshold_name'],'hits':k,'samples':N,'exact_p':p,'expected_hits':N*p,'z_binomial':score,'exact_two_sided_binomial_p':pval})
assert [x['hits'] for x in old_summary]==[637,4395,94,852]
for j,seed in enumerate((20261018,20261019)):
    rr=old[2*j:2*j+2];gen=np.random.Generator(np.random.PCG64(seed));hits=[0,0];N=10_000_000
    for _ in range(20):
        trial=gen.uniform(0,rr[0]['upper'],500_000)
        for k,row in enumerate(rr):
            a,b=row['acceptance_alpha_interval'];hits[k]+=int(np.count_nonzero((trial>=a)&(trial<=b)))
    assert hits==([[332,2226],[58,395]][j]),(seed,hits)
    new_summary.append({'seed':seed,'samples':N,'hits':hits,'exact_expectations':[N*(r['acceptance_alpha_interval'][1]-r['acceptance_alpha_interval'][0])/r['upper'] for r in rr]})
record={'status':'PASS_WITH_DISCLOSED_RARE_EVENT_FLUCTUATION',
        'source_whitening':{'lower_gram_eigenvalue':g,'gram_delta':d,'embedding_error':v_error,'embedding_upper_bound':v_upper},
        'relative_moments':checks,'toeplitz_min_eig':eig_min,
        'exterior_schur':{'z_real':z.real,'z_imag':z.imag,'K':K,'actual_error':actual,'certified_upper':upper},
        'rare_event_sampling':{'original_counts':old_summary,'new_seed_followup':new_summary,'note':'Counts are nested within each generator seed. Binomial diagnostics refer to arbitrary declared uniform-alpha measures, not to the evidence for a physical theory.'},
        'scope':'Finite-matrix validation of general norm inequalities; cannot certify infinite-volume limits, physical mass poles, or gravitational dynamics.'}
(OUT/'QUANTITATIVE_STABILITY_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
