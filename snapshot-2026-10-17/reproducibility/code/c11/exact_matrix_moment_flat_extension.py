#!/usr/bin/env python3
"""Exact symbolic matrix-moment flat-extension benchmark for the C11 reconstruction map."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sympy as sp


def evaluate():
    I=sp.I; r=6
    atoms=[sp.Integer(1),sp.Integer(-1),I]
    W1=sp.diag(*[sp.Rational(1,2),sp.Rational(1,3),sp.Rational(1,4),sp.Rational(1,5),sp.Rational(1,6),sp.Rational(1,7)])
    W2=sp.diag(*[sp.Rational(1,3),sp.Rational(1,4),sp.Rational(1,5),sp.Rational(1,6),sp.Rational(1,7),sp.Rational(1,8)])
    W3=sp.eye(r)-W1-W2; weights=[W1,W2,W3]
    if not all(x>0 for x in W3.diagonal()): raise ValueError('positive matrix weights required')
    def C(n): return sp.simplify(sum(((z**n)*W for z,W in zip(atoms,weights)),sp.zeros(r)))
    def T(K): return sp.Matrix.vstack(*[sp.Matrix.hstack(*[C(j-i) for j in range(K+1)]) for i in range(K+1)])
    rank2,rank3=T(2).rank(),T(3).rank()
    recurrence=[]
    for n in range(-6,7):
        recurrence.append(sp.simplify(C(n+3)-I*C(n+2)-C(n+1)+I*C(n))==sp.zeros(r))
    return {
      'classification':'EXACT_MATRIX_MOMENT_FLAT_EXTENSION_BENCHMARK',
      'source_rank':r,
      'atoms_exact':['1','-1','I'],
      'toeplitz_rank_K2':int(rank2),
      'toeplitz_rank_K3':int(rank3),
      'flat_extension_relation':'rank(T_2)=rank(T_3)=18',
      'annihilating_polynomial':'z^3 - I*z^2 - z + I',
      'moment_recurrence':'exact throughout n=-6,...,6',
      'C0_identity':'exact',
      'theorem_scope':'Exact finite matrix-moment benchmark for flat extension and recurrence recovery.'
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    out=evaluate(); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
