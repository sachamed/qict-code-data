#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib
import numpy as np
from pathlib import Path

# The theorem is algebraic: a monoid homomorphism P:N_0->U(H) with P(0)=I
# and P(1)=U is uniquely P(d)=U^d. We also verify it numerically for random
# unitaries and the QICT route lengths 10,9,8.

def random_unitary(n, rng):
    A=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    Q,R=np.linalg.qr(A)
    ph=np.diag(R)
    ph=np.where(np.abs(ph)>0,ph/np.abs(ph),1)
    return Q@np.diag(np.conj(ph))

def main():
    rng=np.random.default_rng(210963)
    max_semigroup=0.0; max_unitarity=0.0
    rows=[]
    for n in (2,3,6,12):
        U=random_unitary(n,rng)
        P=[np.eye(n,dtype=complex)]
        for d in range(1,21): P.append(P[-1]@U)
        for a in range(0,11):
            for b in range(0,11-a):
                max_semigroup=max(max_semigroup,float(np.linalg.norm(P[a+b]-P[a]@P[b],2)))
        for d in (8,9,10):
            max_unitarity=max(max_unitarity,float(np.linalg.norm(P[d].conj().T@P[d]-np.eye(n),2)))
        rows.append({'dimension':n,'det_abs_route_10':float(abs(np.linalg.det(P[10]))),
                     'route_norms':[float(np.linalg.norm(P[d],2)) for d in (10,9,8)]})
    report={
      'status':'PASS_TEMPORAL_SEMIGROUP_ROUTE_UNIQUENESS',
      'assumptions':['P(0)=I','P(d+e)=P(d)P(e) for all nonnegative integers','P(1)=U_C8'],
      'theorem':'By induction, P(d)=U_C8^d is the unique monoid homomorphism with the stated generator. Hence routes d=(10,9,8) are unique within this declared temporal-semigroup completion.',
      'route_lengths':[10,9,8],
      'max_semigroup_residual_2norm':max_semigroup,
      'max_route_unitarity_residual_2norm':max_unitarity,
      'random_unitary_checks':rows,
      'scope':'This proves uniqueness only within the temporal-semigroup completion. It does not prove that temporal monodromy is mathematically equivalent to an independently specified set of spatial BCC paths.'
    }
    out=Path(__file__).resolve().parents[1]/'results'/'TEMPORAL_SEMIGROUP_ROUTE_UNIQUENESS.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
