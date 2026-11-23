#!/usr/bin/env python3
"""Finite-dimensional numerical checks of the precise Schur resolvent hypotheses.
The analytic proof is in Supplementary Information; these are finite diagnostics only.
"""
import json
import numpy as np
from pathlib import Path


def check():
    delta=2.0
    beta=0.75
    indefinite=np.diag([-delta,delta])
    B=np.array([[beta,0.0]])
    E=0.0
    counter=float((B@np.linalg.solve(indefinite-E*np.eye(2), B.T))[0,0])
    assert abs(counter+beta**2/delta)<1e-15 and counter<0
    rng=np.random.default_rng(20261122)
    max_norm_margin=0.0;min_positive_eig=float('inf');count=0
    for positive in (False,True):
        for _ in range(100):
            d=np.diag([3.,4.,5.]) if positive else np.diag([-5.,-3.,4.])
            q,_=np.linalg.qr(rng.standard_normal((3,3)))
            M=q@d@q.T
            B=rng.standard_normal((2,3))
            b=np.linalg.norm(B,ord=2)
            e=0.75
            if positive:
                gap=3.;rhs=b*b/(gap-e)
            else:
                gap=3.;rhs=b*b/(gap-abs(e))
            S=B@np.linalg.solve(M-e*np.eye(3),B.T)
            eig=np.linalg.eigvalsh(S)
            assert np.linalg.norm(S,ord=2)<=rhs+1e-10
            max_norm_margin=max(max_norm_margin,float(rhs-np.linalg.norm(S,ord=2)))
            if positive:
                assert eig.min()>=-1e-10 and eig.max()<=rhs+1e-10
                min_positive_eig=min(min_positive_eig,float(eig.min()))
            count+=1
    return {'status':'PASS_FINITE_SCHUR_CONDITIONS','scope':'Finite-dimensional checks, separate from analytic spectral-theorem proof and physical QICT mirror-sector decoupling',
      'indefinite_counterexample_value':counter,'indefinite_counterexample_exact_expression':'-b^2/Delta',
      'operator_order_rejected_for_indefinite_block':True,'operator_order_verified_under_M_greater_equal_Delta_I':True,
      'random_matrix_cases_checked':count,'seed':20261122,
      'positive_case_min_observed_eigenvalue':min_positive_eig,
      'full_physical_mirror_gap_proven':False}

if __name__=='__main__':
    r=check()
    p=Path(__file__).with_name('SCHUR_RESOLVENT_VERIFICATION.json')
    p.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
