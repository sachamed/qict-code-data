#!/usr/bin/env python3
"""Exact symbolic self-test for the matrix moment flat-extension theorem.
This is an algorithmic theorem test, not a physical QICT mass calculation.
"""
import json
from pathlib import Path
import sympy as sp

I = sp.I
r = 6
atoms = [sp.Integer(1), sp.Integer(-1), I]
w1 = [sp.Rational(1,2),sp.Rational(1,3),sp.Rational(1,4),sp.Rational(1,5),sp.Rational(1,6),sp.Rational(1,7)]
w2 = [sp.Rational(1,3),sp.Rational(1,4),sp.Rational(1,5),sp.Rational(1,6),sp.Rational(1,7),sp.Rational(1,8)]
W1 = sp.diag(*w1)
W2 = sp.diag(*w2)
W3 = sp.eye(r)-W1-W2
weights = [W1,W2,W3]
assert all(x > 0 for x in W3.diagonal())

def C(n):
    return sp.simplify(sum(((z**n)*W for z,W in zip(atoms,weights)), sp.zeros(r)))

def toeplitz(K):
    return sp.Matrix.vstack(*[
        sp.Matrix.hstack(*[C(j-i) for j in range(K+1)])
        for i in range(K+1)
    ])

T2, T3 = toeplitz(2), toeplitz(3)
rank2, rank3 = T2.rank(), T3.rank()
# annihilating polynomial p(z)=(z-1)(z+1)(z-i)=z^3-i z^2-z+i
recurrence_zero = True
for n in range(-6,7):
    R = sp.simplify(C(n+3)-I*C(n+2)-C(n+1)+I*C(n))
    recurrence_zero &= (R == sp.zeros(r))

out = {
    "scope":"exact symbolic algorithmic self-test; not a physical mass result",
    "source_rank":r,
    "atoms_exact":["1","-1","I"],
    "toeplitz_rank_K2":rank2,
    "toeplitz_rank_K3":rank3,
    "flat_extension_at_K3": bool(rank3 == rank2),
    "annihilating_polynomial":"z^3 - I*z^2 - z + I",
    "moment_recurrence_exact_zero": bool(recurrence_zero),
    "C0_is_identity": bool(C(0)==sp.eye(r)),
    "mass_extraction_stage":"CERTIFIED_LARGE_VOLUME_C10_KATO_EXECUTION_STAGE",
    "status":"PASS_EXACT_SYMBOLIC_MATRIX_MOMENT_FLAT_EXTENSION_SELFTEST" if rank3==rank2 and recurrence_zero else "FAIL"
}
path=Path(__file__).resolve().parents[2]/"results"/"EXACT_MATRIX_MOMENT_FLAT_EXTENSION_SELFTEST.json"
path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
