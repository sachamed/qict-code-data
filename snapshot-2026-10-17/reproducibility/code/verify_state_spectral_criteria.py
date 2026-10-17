#!/usr/bin/env python3
"""Exact finite checks of spectral state criteria and reference responses.

The rational examples test theorems, not the interacting C10 contraction.
Cyclotomic arithmetic uses the frozen QICT phase exp(-i*pi/63).
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def trim(p):
    p = list(map(F, p))
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def add(p, q):
    return trim([(p[i] if i < len(p) else 0) +
                 (q[i] if i < len(q) else 0)
                 for i in range(max(len(p), len(q)))])


def scale(p, a):
    return trim([a * x for x in p])


def mul(p, q):
    r = [F(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return trim(r)


def divmod_poly(p, q):
    p, q = trim(p), trim(q)
    assert q != [0]
    quotient = [F(0)] * max(1, len(p) - len(q) + 1)
    while p != [0] and len(p) >= len(q):
        k, c = len(p) - len(q), p[-1] / q[-1]
        quotient[k] += c
        p = add(p, [0] * k + scale(q, -c))
    return trim(quotient), p


@lru_cache(None)
def cyclotomic(n):
    p = [-1] + [0] * (n - 1) + [1]
    for d in range(1, n):
        if n % d == 0:
            p, remainder = divmod_poly(p, cyclotomic(d))
            assert remainder == [0]
    assert all(c.denominator == 1 for c in p)
    return tuple(p)


def inverse_mod(p, modulus):
    a, b = trim(modulus), trim(p)
    s0, s1 = [F(0)], [F(1)]
    while b != [0]:
        q, r = divmod_poly(a, b)
        a, b = b, r
        s0, s1 = s1, add(s0, scale(mul(q, s1), -1))
    assert len(a) == 1 and a[0] != 0
    result = divmod_poly(scale(s0, 1 / a[0]), modulus)[1]
    assert divmod_poly(mul(p, result), modulus)[1] == [1]
    return result


def matrix_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def trace_product(a, b):
    return sum(matrix_mul(a, b)[i][i] for i in range(len(a)))


def as_rationals(p):
    return [str(x) for x in p]


def main():
    rho_plus = [[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]]
    rho_minus = [[F(1, 2), F(-1, 2)], [F(-1, 2), F(1, 2)]]
    assert rho_plus != rho_minus
    assert matrix_mul(rho_plus, rho_plus) == rho_plus
    assert matrix_mul(rho_minus, rho_minus) == rho_minus
    assert matrix_mul(rho_plus, rho_minus) == [[0, 0], [0, 0]]
    moment_rows = []
    for n in range(13):
        charged_power = [[F(1), F(0)], [F(0), F((-1) ** n)]]
        c_plus = trace_product(rho_plus, charged_power)
        c_minus = trace_product(rho_minus, charged_power)
        assert c_plus == c_minus == F(1 + (-1) ** n, 2)
        moment_rows.append([n, str(c_plus), str(c_minus)])

    # With a charged identity update, equal source-sector measures coexist
    # with distinct stationary relative moments if neutral phases differ.
    relative_rows = [[n, 1, (-1) ** n] for n in range(13)]
    assert all(row[1] == row[2] for row in relative_rows if row[0] % 2 == 0)
    assert all(row[1] != row[2] for row in relative_rows if row[0] % 2 == 1)

    # One-mode CAR ordering, checked after removing the common phase z**n.
    annihilator = [[F(0), F(1)], [F(0), F(0)]]
    creator = [[F(0), F(0)], [F(1), F(0)]]
    current = [[F(0), F(0)], [F(0), F(1)]]
    vacuum = [[F(1), F(0)], [F(0), F(0)]]
    g2 = trace_product(vacuum, matrix_mul(annihilator, creator))
    g3 = trace_product(vacuum, matrix_mul(matrix_mul(annihilator, current), creator))
    reverse = trace_product(vacuum, matrix_mul(creator, annihilator))
    assert g2 == g3 == 1 and reverse == 0
    assert trace_product(vacuum, current) == 0

    # x=exp(-i*pi/63) has cyclotomic polynomial Phi_126. Division is exact.
    modulus = cyclotomic(126)
    assert len(modulus) - 1 == 36
    assert divmod_poly([-1] + [0] * 125 + [1], modulus)[1] == [0]
    responses = {}
    for cost, amplitude in [(8, 2), (12, 16)]:
        denominator = [1] + [0] * (cost - 1) + [-1]
        coefficient = scale(inverse_mod(denominator, modulus), F(amplitude))
        assert divmod_poly(mul(denominator, coefficient), modulus)[1] == [F(amplitude)]
        responses[str(cost)] = {
            'magnetic_amplitude_magnitude': amplitude,
            'coefficient_divided_by_i_phi': as_rationals(coefficient),
            'linearized_residual_polynomial': ['0'],
            'denominator_invertible_in_Q_x_mod_Phi126': True,
        }

    rank_rows = []
    for m in [1, 2, 4, 8, 16]:
        radius = 63 * m
        labels = [r for r in range(-radius, radius + 1)
                  if (8 * r * r) % 126 == 0]
        assert labels == list(range(-radius, radius + 1, 21))
        assert len(labels) == 6 * m + 1
        distance = 1 - F(6 * m + 1, 12 * m + 1)
        rank_rows.append({'radius': radius, 'rank': len(labels),
                          'distance_to_doubled_radius': str(distance)})

    certificate = {
        'status': 'PASS_RATIONAL_STATE_SPECTRAL_CRITERIA',
        'arithmetic': 'fractions.Fraction; exact polynomial Euclidean division',
        'source_program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Finite algebraic criteria and the one-plaquette electric reference; not a moving C10 contraction.',
        'state_measure_example': {
            'distinct_orthogonal_source_states': True,
            'same_spectral_weights': ['1/2', '1/2'],
            'all_orders_formula': '(1+(-1)**n)/2',
            'checked_moments': moment_rows,
        },
        'stationary_relative_phase_example': {
            'neutral_eigenvalues': [1, -1], 'charged_update': 'I_2',
            'source_sector_moments': '1 for both states',
            'relative_moments': relative_rows,
        },
        'one_mode_CAR_ordering': {
            'two_point_coefficient': str(g2),
            'connected_single_current_coefficient': str(g3),
            'opposite_two_point_coefficient': str(reverse),
            'common_phase': 'exp(-i*0.7*n)',
        },
        'frozen_phase_cyclotomic_polynomial': as_rationals(modulus),
        'nonresonant_reference_responses': responses,
        'electric_Floquet_seed_ranks': rank_rows,
        'publication_sources': ['paper/sections/supplement/s08.tex',
                                'paper/sections/supplement/s55.tex',
                                'paper/sections/supplement/s61.tex',
                                'paper/sections/main/m101.tex'],
    }
    certificate['publication_sources_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in certificate['publication_sources']
    }
    output = ROOT / 'reproducibility/results/spectral_state_criteria'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'STATE_SPECTRAL_CRITERIA_CERTIFICATE.json'
    path.write_text(json.dumps(certificate, indent=2) + '\n')
    print(json.dumps({'status': certificate['status'],
                      'arithmetic': certificate['arithmetic'],
                      'response_costs': list(responses),
                      'source_state_moment_tests': len(moment_rows),
                      'certificate': str(path.relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    main()
