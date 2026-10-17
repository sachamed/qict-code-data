#!/usr/bin/env python3
"""Finite-regulator positive tensor mass-squared spectral-weight certificate.

The generated four-state matrices are mathematical test instances, NOT QICT
interacting C10 operators. --input-npz accepts an external square Hermitian
K (a physical mass-squared operator), an n-by-2 J (physical TT sources), and
(optional) epsilon, s.  It never constructs a positive generator from a
Floquet unitary or identifies a static susceptibility with a propagator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

TOL=3e-11

def source_certificate(K, J, epsilon, s):
    K=np.asarray(K,complex); J=np.asarray(J,complex)
    n=K.shape[0]
    assert K.shape==(n,n) and J.shape==(n,2) and n>=2
    assert np.all(np.isfinite(K)) and np.all(np.isfinite(J))
    herm=np.linalg.norm(K-K.conj().T,2)
    assert herm < TOL*max(1,np.linalg.norm(K,2)), 'non-Hermitian K'
    E,V=np.linalg.eigh((K+K.conj().T)/2)
    assert E.min()>=-TOL, 'negative squared mass'
    assert epsilon>0 and s>0
    E=np.maximum(E,0.)
    A=V.conj().T@J
    weights=np.einsum('ni,nj->nij', A.conj(),A) # each rank-one 2x2
    lo=E<=epsilon
    assert np.any(lo) and np.any(~lo), 'need low and complementary bands'
    # Projector orientation check: rows A[j,i], b_j^dagger b_j
    W=np.sum(weights[lo],axis=0)
    D=np.einsum('n,nij->ij', 1/(E+s),weights)
    M=float(np.linalg.norm(J,2)**2)
    g=float(np.min(E[~lo]))
    assert g>epsilon
    Z=float(np.trace(W).real/2)
    eta=float(np.linalg.norm(W-Z*np.eye(2),2))
    err=float(np.linalg.norm(s*D-Z*np.eye(2),2))
    bound=float(eta+M*epsilon/s+M*s/g)
    assert err<=bound+2e-12, ('spectral bound violated',err,bound)
    assert np.linalg.eigvalsh(W).min()>-TOL
    assert np.linalg.eigvalsh(D).min()>-TOL
    # direct eigenvector reconstruction test
    D_ref=J.conj().T@np.linalg.inv(K+s*np.eye(n))@J
    assert np.linalg.norm(D-D_ref,2)<5e-11
    return {'dimension':n,'low_eigenvalues':[float(x) for x in E[lo]],
      'complement_gap':g,'low_weight_matrix':np.real_if_close(W).tolist(),
      'polarization_weights':[float(x) for x in np.linalg.eigvalsh(W)],
      'source_norm_bound_M':M,'isotropic_target_Z_from_low_weight':Z,
      'polarization_error_eta':eta,'scale_s':s,'epsilon':epsilon,
      'absolute_residue_error':err,'certified_upper_bound':bound,'pass':True}

def coupled_example(n):
    # Unitary rotations couple both physical sources to their environments.
    th=.37+.05/n; c=np.cos(th); sn=np.sin(th)
    V=np.eye(4)
    for low,high in [(0,2),(1,3)]:
        V[low,low]=c;V[high,low]=sn
        V[low,high]=-sn;V[high,high]=c
    ev=np.array([1/n**2,(1+.1/n)/n**2,1.,1.5])
    K=V@np.diag(ev)@V.T
    J=np.eye(4)[:,:2]
    eps=2/n**2
    s=np.sqrt(eps)
    return source_certificate(K,J,eps,s)

def vanishing_weight_example(n):
    K=np.diag([n**-2,1.])
    J=np.array([[1/n],[np.sqrt(1-n**-2)]])
    return {'n':n,'lowest_squared_mass':1/n**2,
     'source_low_weight':1/n**2,
     'response_at_s_equal_1':float((J.T@np.linalg.inv(K+np.eye(2))@J)[0,0])}


def random_matrix_stress_test():
    """Deterministic complex-Hermitian tests of the spectral bound."""
    rng=np.random.default_rng(4814)
    max_err=0.; min_margin=float('inf')
    for _ in range(128):
        n=8
        A=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        Q,_=np.linalg.qr(A)
        eigenvalues=np.sort(np.r_[rng.uniform(0,0.0005,size=2),rng.uniform(1,3,size=n-2)])
        K=Q@np.diag(eigenvalues)@Q.conj().T
        J=rng.normal(size=(n,2))+1j*rng.normal(size=(n,2))
        J=J/np.linalg.norm(J,2)
        result=source_certificate(K,J,epsilon=.001,s=.05)
        max_err=max(max_err,result['absolute_residue_error'])
        min_margin=min(min_margin,result['certified_upper_bound']-result['absolute_residue_error'])
    assert min_margin>-2e-12
    return {'repetitions':128,'rng':'numpy.PCG64','seed':4814,
            'minimum_bound_margin':min_margin,'maximum_observed_residue_error':max_err,'status':'PASS'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-npz',type=Path,help='external K,J with operator provenance')
    p.add_argument('--epsilon',type=float)
    p.add_argument('--s',type=float)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.input_npz:
        assert a.epsilon and a.s
        d=np.load(a.input_npz,allow_pickle=False)
        input_hash=hashlib.sha256(a.input_npz.read_bytes()).hexdigest()
        result={'classification':'EXTERNAL_OPERATOR_FINITE_SPECTRAL_DIAGNOSTIC',
          'operator_origin':'user-supplied; not independently identified as QICT',
          'input_sha256':input_hash,
          'certificate':source_certificate(d['K'],d['J'],a.epsilon,a.s)}
    else:
        sizes=[8,16,32,64,128,256]
        tested=[coupled_example(n) for n in sizes]
        for n, cert in zip(sizes,tested):
            # Exact finite-volume residue is cos^2(theta_n).
            assert abs(cert['isotropic_target_Z_from_low_weight']-np.cos(.37+.05/n)**2)<TOL
            assert cert['polarization_error_eta']<1e-10
        vanishing=[vanishing_weight_example(n) for n in sizes]
        assert vanishing[-1]['source_low_weight']<vanishing[0]['source_low_weight']
        assert abs(vanishing[-1]['response_at_s_equal_1']-0.5)<abs(vanishing[0]['response_at_s_equal_1']-0.5)
        result={'classification':'SYNTHETIC_POSITIVE_OPERATOR_BENCHMARK',
          'physical_qict_tensor_operator_supplied':False,
          'main_theorem':'two-polarization spectral-projection enclosure',
          'coupled_positive_matrix_checks':tested,
          'vanishing_weight_counterexample':vanishing,
          'target_limiting_residue_from_test_matrices':float(np.cos(.37)**2),
          'verification':'All finite-regulator bounds satisfied; no QICT graviton claim.',
          'random_complex_matrix_stress_test':random_matrix_stress_test()}
    serialized=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(serialized)
    print(serialized)

if __name__=='__main__': main()
