#!/usr/bin/env python3
"""Independent counterexamples and fixed-seed reduced-law sampling checks.

No experimental data are used to infer a physical pole or certify a theory.
"""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'reproducibility/results/structural_corrections'
DEST.mkdir(parents=True,exist_ok=True)

def test_pointed_rank():
    s=np.eye(7)
    for j in (0,2,4):
        s[j,j]=s[j+1,j+1]=0
        s[j,j+1]=s[j+1,j]=1
    assert np.array_equal(s@s,np.eye(7))
    assert abs(np.trace(s)-1)<1e-12
    J=np.block([[np.zeros((7,7)),np.eye(7)],[np.eye(7),np.zeros((7,7))]])
    T=np.hstack([np.eye(7),np.eye(7)])
    assert np.array_equal(T@J,T)
    assert np.linalg.norm(T)>0
    return {'pointed_fixed_anchor':True,'nonzero_even_sheet_intertwiner':True,'rank_odd_involution_fixed_point':True}

def test_relative():
    U0=np.eye(2)
    Uf=np.diag([1.,-1.])
    v=np.array([np.sqrt(3)/2,.5])
    X=np.outer(v,[1.,0.])
    W=np.kron(np.conj(U0),Uf)
    x=X.reshape(-1,order='F')
    c0=x.conj()@x; c1=x.conj()@W@x
    assert abs(c0-1)<1e-14 and abs(c1-.5)<1e-14
    assert abs(v.conj()@Uf.T@Uf@v-1)<1e-14
    C=lambda n: x.conj()@np.linalg.matrix_power(W,n)@x
    Toeplitz=np.array([[C(k-j) for k in range(6)] for j in range(6)])
    assert np.linalg.eigvalsh(Toeplitz).min()>-1e-12
    return {'C0':float(c0),'C1':float(c1),'incorrect_unitarity_substitution':1,'Toeplitz_min_eig':float(np.linalg.eigvalsh(Toeplitz).min())}

def test_schur():
    z=1.7+0.1j;result=[]
    for L in (8,11,19):
        U=np.roll(np.eye(L),1,axis=0)
        C=[np.linalg.matrix_power(U,n)[0,0] for n in range(7)]
        assert C==[1,0,0,0,0,0,0]
        G=np.linalg.inv(z*np.eye(L)-U)[0,0]
        sigma=z-1/G
        assert abs(sigma-z**(1-L))<1e-11
        result.append({'cycle_length':L,'first_six_moments':[float(v) for v in C[1:]],'sigma_abs':float(abs(sigma)),'identity_residual':float(abs(sigma-z**(1-L)))})
    return {'cycles':result,'bilateral_shift_sigma':0.0,'fixed_low_moments_do_not_determine_memory':True}

def test_monte_carlo():
    p=ROOT/'reproducibility/results/reduced_hierarchy_tests/REDUCED_HIERARCHY_TESTS.json'
    data=json.loads(p.read_text());res=[]
    for j in range(2):
        rows=data['Monte_Carlo'][2*j:2*j+2]
        rng=np.random.Generator(np.random.PCG64(20261014+j))
        k=[0,0]
        for _ in range(40):
            samples=rng.uniform(0,rows[0]['upper'],500000)
            for i,row in enumerate(rows):
                a,b=row['acceptance_alpha_interval']
                k[i]+=int(np.count_nonzero((samples>=a)&(samples<=b)))
        res.extend(k)
    assert res==[637,4395,94,852]
    return {'replayed_counts':res,'expected_counts':[637,4395,94,852],'sampling_algorithm':'NumPy PCG64; seeds 20261014, 20261015; 20 million draws each','input_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

if __name__=='__main__':
    out={'status':'PASS','pointed_opposition':test_pointed_rank(),'relative_gns':test_relative(),'feshbach_complement':test_schur(),'monte_carlo':test_monte_carlo()}
    (DEST/'STRUCTURAL_CORRECTIONS_VERIFIED.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
