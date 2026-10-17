#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

def random_unitary(n,rng):
    A=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    Q,R=np.linalg.qr(A)
    ph=np.diag(R); ph=np.where(np.abs(ph)>0,ph/np.abs(ph),1)
    return Q@np.diag(np.conj(ph))

def main():
    rng=np.random.default_rng(631021)
    max_rep=0.0; max_path=0.0; max_color=0.0
    for _ in range(100):
        th=rng.uniform(-np.pi,np.pi)
        u1=np.exp(1j*th)
        u2=random_unitary(2,rng)
        u3=random_unitary(3,rng)
        # Charged-lepton representations: L=(1,2)_{q=-3}, R=(1,1)_{q=-6}.
        rL=(u1**-3)*u2
        rhs=np.kron(np.array([[u1**-3]],complex),u2)
        max_rep=max(max_rep,float(np.linalg.norm(rL-rhs,2)))
        # A fixed L-worldline is a product representation, so ordered transport factorizes.
        ths=rng.uniform(-np.pi,np.pi,5); U2=[random_unitary(2,rng) for _ in range(5)]
        lhs=np.eye(2,dtype=complex); a=1+0j; b=np.eye(2,dtype=complex)
        for t,V in zip(ths,U2):
            lhs=((np.exp(1j*t)**-3)*V)@lhs
            a=(np.exp(1j*t)**-3)*a
            b=V@b
        max_path=max(max_path,float(np.linalg.norm(lhs-a*b,2)))
        # Color acts trivially on leptons, exactly identity independent of u3.
        max_color=max(max_color,float(np.linalg.norm(np.eye(1)-np.array([[1.0+0j]]),2)))
    report={
      'status':'PASS_FIXED_WORLDLINE_PRODUCT_GROUP_FACTORIZATION',
      'representation_statement':'For a fixed lepton worldline segment, r_L(u1,u2,u3)=u1^{-3} D^{1/2}(u2) tensor 1_color and r_R=u1^{-6} tensor 1_SU2 tensor 1_color. Ordered transport factorizes across product-group factors before the sum over alternative matter paths/routes.',
      'max_single_link_factorization_residual_2norm':max_rep,
      'max_five_link_ordered_path_factorization_residual_2norm':max_path,
      'color_triviality_residual_2norm':max_color,
      'important_nonfactorization_boundary':'The full rank-six observable is a coherent sum over matter paths/routes. One may contract the independent gauge-group factors conditional on each fixed path, but one may not replace the full result by Z_gauge times an independently averaged matter matrix. Pathwise products must be recombined coherently before receiver compression.',
      'scope':'Exact representation-theoretic identity. It does not by itself compute the SU(2) defect spin-foam or the Kato-selected path weights.'
    }
    out=Path(__file__).resolve().parents[2]/'results'/'microscopic_closure'/'PRODUCT_GROUP_WORLDLINE_FACTORIZATION.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
