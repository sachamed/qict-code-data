#!/usr/bin/env python3
"""Deterministic certificate for the discrete rigidity statements added to C8.

This check does not establish the physical correctness of C8. It verifies two
algebraic consequences of the declared primitive class: (i) cyclic changes of
stroboscopic origin are unitary conjugacies of the full ordered Floquet product,
and (ii) the shortest positive faithful character of Z_126 is uniquely k=1.
"""
from __future__ import annotations
import json
from math import gcd, pi
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'certificates' / 'c8_macro_rigidity_certificate.json'


def random_unitary(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    phases = np.where(np.abs(d) > 0, d / np.abs(d), 1.0)
    return q @ np.diag(np.conj(phases))


def main() -> None:
    seed = 20260909
    rng = np.random.default_rng(seed)
    # Rightmost factor acts first. The ordered classes are
    # stream, cell, electric, plaquette, internal.
    factors = [random_unitary(rng, 6) for _ in range(5)]
    stream, cell, electric, plaquette, internal = factors
    U = internal @ plaquette @ electric @ cell @ stream

    # Move the rightmost (earliest) stream layer to the left. This is a cyclic
    # stroboscopic-origin change: U' = stream U stream^†.
    U_shift = stream @ internal @ plaquette @ electric @ cell
    cyclic_defect = float(np.linalg.norm(U_shift - stream @ U @ stream.conj().T, 2))

    # Enumerate all positive principal faithful characters of Z_126.
    faithful = [k for k in range(1, 64) if gcd(k, 126) == 1]
    phases = {k: 2 * pi * k / 126 for k in faithful}
    min_phase = min(phases.values())
    minimizers = [k for k, v in phases.items() if abs(v - min_phase) < 1e-15]

    cert = {
        'status': 'PASS',
        'task': 'C8 macro-step rigidity',
        'seed': seed,
        'matrix_dimension': 6,
        'declared_positive_order': ['stream', 'cell', 'electric', 'plaquette', 'internal'],
        'cyclic_time_origin_conjugacy_defect': cyclic_defect,
        'faithful_positive_characters_Z126': faithful,
        'shortest_positive_faithful_k': minimizers,
        'selected_k': 1,
        'selected_phase_exact': 'pi/63',
        'selected_phase_numeric': phases[1],
        'statement': (
            'Within the declared C8 ordering rule, cyclic time-origin shifts are '
            'unitary conjugacies and the shortest positive faithful Z_126 character '
            'is uniquely k=1. Noncyclic orderings are excluded by the C8 postulate, '
            'not derived by this numerical certificate.'
        ),
    }
    assert cyclic_defect < 1e-12
    assert minimizers == [1]
    OUT.write_text(json.dumps(cert, indent=2) + '\n')
    print('PASS C8-macro-rigidity', json.dumps(cert, indent=2))


if __name__ == '__main__':
    main()
