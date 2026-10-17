#!/usr/bin/env python3
"""Exact checks for the 17 October targeted spectral-domain correction.

Run from any directory. These finite identities check a Cayley sign and
the domain of a Schur inverse; they do not certify physical matching.
"""
from pathlib import Path
import json
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproducibility/results/quick_spectral_domain_checks'


def main():
    t = sp.symbols('t', real=True)
    u = (1 + sp.I*t)/(1 - sp.I*t)
    assert sp.simplify(sp.I*(1-u)/(1+u)-t) == 0
    assert sp.simplify(sp.I*(1-sp.I)/(1+sp.I)) == 1

    # U has eigenvalues +/- i; the complementary compression D is zero.
    # At z=0 the full resolvent exists but its scalar P-compression is zero.
    unitary = sp.Matrix([[0, 1], [-1, 0]])
    assert unitary.H*unitary == sp.eye(2)
    assert unitary.det() == 1
    assert (-unitary).inv()[0, 0] == 0
    z = sp.symbols('z')
    green = (z*sp.eye(2)-unitary).inv()[0, 0]
    schur = z+1/z
    assert sp.simplify(green*schur-1) == 0
    assert green.subs(z, 2) == sp.Rational(2, 5)
    assert schur.subs(z, 2) == sp.Rational(5, 2)

    result = {
        'status': 'PASS',
        'scope': 'Exact finite algebra; physical and continuum matching remain separate.',
        'cayley_rational_identity': 'i*(1-u)/(1+u)=t for u=(1+i*t)/(1-i*t)',
        'cayley_at_eigenvalue_i': 1,
        'full_resolvent_exists_at_zero': True,
        'compressed_resolvent_at_zero': 0,
        'complementary_resolvent_exists_at_zero': False,
        'schur_identity_on_common_domain': 'z/(z^2+1) * (z+1/z) = 1',
        'common_domain_example_z_2': 'G=2/5; Gamma=5/2'
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'CHECKS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
