#!/usr/bin/env python3
"""Finite matrix checks for overlap-gap and positive spectral-weight theorems.

Mathematical test carriers only. Does not evaluate the physical BCC gauge-field
path integral, a gravitational propagator, or a physical particle mass.
"""
import json
from pathlib import Path
import numpy as np


def matrix_sign(h):
    vals, vecs = np.linalg.eigh(h)
    assert np.min(np.abs(vals)) > 0
    return (vecs * np.sign(vals)) @ vecs.conj().T


def main():
    gen = np.random.default_rng(20400617)
    a = gen.normal(size=(6, 6)) + 1j * gen.normal(size=(6, 6))
    q, _ = np.linalg.qr(a)
    gamma5 = np.diag([1, 1, 1, -1, -1, -1]).astype(complex)
    h = q @ np.diag([-2.1, -1.3, -0.9, 0.8, 1.4, 2.4]) @ q.conj().T
    h = (h + h.conj().T)/2
    gamma = np.min(np.abs(np.linalg.eigvalsh(h)))
    e = gen.normal(size=(6, 6)) + 1j * gen.normal(size=(6, 6))
    e = (e + e.conj().T)/2
    e *= 0.13 / np.linalg.norm(e, 2)
    eps = np.linalg.norm(e, 2)
    hp = h + e
    s, sp = matrix_sign(h), matrix_sign(hp)
    d, dp = np.eye(6) + gamma5 @ s, np.eye(6) + gamma5 @ sp
    overlap_gap_error = float(np.linalg.norm(sp - s, 2))
    overlap_upper = float(eps / (gamma - eps))
    proj_error = overlap_gap_error / 2
    assert overlap_gap_error <= overlap_upper * (1 + 1e-12)
    gw_residual = float(np.linalg.norm(gamma5@d + d@gamma5 - d@gamma5@d, 2))
    assert gw_residual < 1e-12
    assert np.linalg.norm(d.conj().T - gamma5@d@gamma5, 2) < 1e-12
    index = 0.5 * np.trace(gamma5-s)
    index_new = 0.5 * np.trace(gamma5-sp)
    assert abs(index - index_new) < 1e-11
    assert abs(index.imag) < 1e-11
    assert abs(index.real - round(index.real)) < 1e-11

    # Positive measure with a massless atom and a strictly massive part.
    w0 = 0.8
    weights = np.array([w0, 0.3, 1.4, 0.2])
    masses2 = np.array([0., 0.15, 2., 12.])
    residues = []
    for s0 in (1e-2, 1e-4, 1e-6):
        val = float(s0 * np.sum(weights / (s0 + masses2)))
        residues.append({'s':s0, 's_chi':val,
                         'difference_from_massless_atom':abs(val - w0)})
    assert residues[2]['difference_from_massless_atom'] < residues[0]['difference_from_massless_atom']
    for sample in (0.01, 0.0001, 0.000001):
        # Certify the theorem's elementary finite-measure inequality directly.
        lowmass = float(sum(w for t, w in zip(masses2, weights) if 0<t<1))
        highmass = float(sum(w for t, w in zip(masses2, weights) if t>=1))
        deviation = float(sample*sum(w/(sample+t) for t,w in zip(masses2[1:],weights[1:])))
        assert deviation <= lowmass + sample*highmass + 1e-12

    out = {
      'scope':'mathematical_float64_checks_not_microscopic_BCC_interacting_completion',
      'overlap':{
        'gap_lower':float(gamma), 'perturbation_norm':float(eps),
        'observed_sign_change':overlap_gap_error,
        'analytical_bound_evaluated_float64':overlap_upper,
        'observed_chiral_projector_change':proj_error,
        'gw_operator_identity_residual':gw_residual,
        'index_start':int(round(index.real)),
        'index_end':int(round(index_new.real))},
      'tt_spectral_example':{'massless_atom_weight':w0, 'infrared_sequence':residues},
      'uses_experimental_inputs':False,
      'result':'PASS'
    }
    dst=Path(__file__).resolve().parents[1]/'results'/'OVERLAP_TENSOR_MATHEMATICAL_CHECKS.json'
    dst.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({'result':'PASS', 'index':out['overlap']['index_start'],
                      'spectral_limit':w0},indent=2))


if __name__ == '__main__':
    main()
