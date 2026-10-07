#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sympy as sp


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    E, Delta = sp.symbols('E Delta', integer=True)
    P=2*E**2+E
    triangular=sp.simplify(P.subs(E,Delta) - (2*Delta)*(2*Delta+1)/2)
    # Hypercharge anomaly identities in left-handed Weyl convention, with right fields conjugated.
    qQ,qu,qd,qL,qe,qnu=1,4,-2,-3,-6,0
    # SU3^2 U1: 2 qQ - qu - qd; SU2^2 U1: 3qQ+qL
    a33=sp.Integer(2*qQ-qu-qd)
    a22=sp.Integer(3*qQ+qL)
    agrav=sp.Integer(6*qQ-3*qu-3*qd+2*qL-qe-qnu)
    acub=sp.Integer(6*qQ**3-3*qu**3-3*qd**3+2*qL**3-qe**3-qnu**3)
    # Weyl relation XZ = omega^{-1} ZX for the conventional clock/shift matrices.
    omega=sp.exp(2*sp.pi*sp.I/3)
    X=sp.Matrix([[0,0,1],[1,0,0],[0,1,0]])
    Z=sp.diag(1,omega,omega**2)
    weyl=sp.simplify(X*Z-omega**(-1)*Z*X)
    # BCC quadratic high-symmetry values.
    def b(nx,ny,nz): return sp.simplify((11-8*(nx**4+ny**4+nz**4))/32)
    b_axis=b(1,0,0); b_face=b(1/sp.sqrt(2),1/sp.sqrt(2),0); b_body=b(1/sp.sqrt(3),1/sp.sqrt(3),1/sp.sqrt(3))
    ratios=[sp.simplify(96*x) for x in [b_axis,b_face,b_body]]
    out={
      'classification':'SYMBOLIC_PROOF_RECONSTRUCTION',
      'primitive_electric_polynomial':str(P),
      'triangular_identity_residual':str(triangular),
      'triangular_values_Delta_1_3_5':[int(P.subs(E,d)) for d in [1,3,5]],
      'hypercharge_anomaly_residuals':{'SU3_SU3_U1':int(a33),'SU2_SU2_U1':int(a22),'grav_grav_U1':int(agrav),'U1_cubic':int(acub)},
      'weyl_relation_max_exact_residual':str(max([sp.simplify(abs(x)) for x in weyl], default=0)),
      'bcc_b_values':{'axis':str(b_axis),'face':str(b_face),'body':str(b_body)},
      'bcc_ratio_after_common_96_factor':[str(x) for x in ratios],
      'all_checks_pass': bool(triangular==0 and a33==0 and a22==0 and agrav==0 and acub==0 and weyl==sp.zeros(3))
    }
    Path(args.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
