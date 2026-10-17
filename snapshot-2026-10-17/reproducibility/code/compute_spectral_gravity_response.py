#!/usr/bin/env python3
"""Finite-regulator QICT extraction of homogeneous and TT metric-source responses.

The input arrays describe one source-deformed regulator update.  The update time
``delta_t_R`` and primitive spatial cell volume ``v_cell_R`` vanish along the
continuum net, whereas the finite QICT metrological scale is ``tau_*`` with
``ell_*=c tau_*``.  Their conversion is supplied dimensionlessly as

    continuum_density_factor = (tau_*/delta_t_R) * (ell_*^3/v_cell_R).

Required NPZ fields
-------------------
K0, K1_vol, K2_qmq                : same-shape Hermitian operator arrays
K1_q, K1_mq                      : real-amplitude Hermitian derivatives, or a Fourier adjoint pair
N_cells                           : number of regulator spatial cells
qell                              : |q| ell_*
continuum_density_factor          : continuum density conversion Z_R

Optional
--------
selected_index : isolated eigenbranch index (default: lowest eigenvalue)
source_basis   : 'real_amplitudes' (default) or 'fourier_pair'

With P0 the selected Riesz projector and
S0=Q0(kappa0-K0)^{-1}Q0, the regulator-cell responses are

    epsilon_cell = Re Tr(P0 K1_vol)/(4 N_cells)
    chi_cell(q)  = Re Tr[P0 (K2_qmq+K1_q S0 K1_mq+K1_mq S0 K1_q)]/N_cells.

The fixed-Copy-Time continuum coordinates are

    epsilon0_src_R = Z_R * epsilon_cell
    chi_hat_TT_R(q) = Z_R * chi_cell(q).

The q=0-subtracted q^2 coefficient of chi_hat supplies gamma_TT^(s).  Its
identification with the Einstein--Hilbert coefficient gamma_R follows on the
canonically normalized Lorentzian dynamical-metric branch specified in the
manuscript.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def hermitian_error(A: np.ndarray) -> float:
    return float(np.linalg.norm(A-A.conj().T, ord='fro'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('input_npz')
    ap.add_argument('-o', '--output', default=None)
    a = ap.parse_args()
    d = np.load(a.input_npz, allow_pickle=False)

    required = ['K0','K1_vol','K1_q','K1_mq','K2_qmq','N_cells','qell','continuum_density_factor']
    missing = [k for k in required if k not in d.files]
    if missing:
        raise SystemExit(f"missing required NPZ fields: {missing}")

    K0 = np.asarray(d['K0'], complex)
    K1v = np.asarray(d['K1_vol'], complex)
    K1q = np.asarray(d['K1_q'], complex)
    K1m = np.asarray(d['K1_mq'], complex)
    K2 = np.asarray(d['K2_qmq'], complex)
    mats = {'K0':K0,'K1_vol':K1v,'K1_q':K1q,'K1_mq':K1m,'K2_qmq':K2}
    if any(A.ndim != 2 or A.shape != K0.shape or not np.all(np.isfinite(A)) for A in mats.values()):
        raise SystemExit('all operator arrays must have the same square shape')
    if K0.shape[0] != K0.shape[1] or K0.shape[0] == 0:
        raise SystemExit('operator arrays must be nonempty and square')
    source_basis = str(np.asarray(d['source_basis']).reshape(())) if 'source_basis' in d.files else 'real_amplitudes'
    if source_basis not in {'real_amplitudes', 'fourier_pair'}:
        raise SystemExit('source_basis must be real_amplitudes or fourier_pair')
    tol = 1e-10
    for name in ('K0', 'K1_vol', 'K2_qmq') + (('K1_q', 'K1_mq') if source_basis == 'real_amplitudes' else ()):
        A = mats[name]
        if hermitian_error(A) > tol * max(1.0, float(np.linalg.norm(A, 'fro'))):
            raise SystemExit(f'{name} must be Hermitian in the declared source basis')
    adjoint_error = float(np.linalg.norm(K1m-K1q.conj().T, 'fro'))
    if source_basis == 'fourier_pair' and adjoint_error > tol * max(1.0, float(np.linalg.norm(K1q, 'fro'))):
        raise SystemExit('Fourier derivatives must satisfy K1_mq = K1_q^dagger')

    Ncells = float(np.asarray(d['N_cells']).reshape(()))
    qell = float(np.asarray(d['qell']).reshape(()))
    ZR = float(np.asarray(d['continuum_density_factor']).reshape(()))
    if not (np.isfinite(Ncells) and Ncells > 0 and Ncells.is_integer()):
        raise SystemExit('N_cells must be a positive integer')
    if not (np.isfinite(qell) and qell >= 0):
        raise SystemExit('qell must be finite and nonnegative')
    if not (ZR > 0 and np.isfinite(ZR)):
        raise SystemExit('continuum_density_factor must be finite and positive')

    w, V = np.linalg.eigh(K0)
    idx = int(np.asarray(d['selected_index']).reshape(())) if 'selected_index' in d.files else int(np.argmin(w))
    if idx < 0 or idx >= len(w):
        raise SystemExit('selected_index out of range')
    k0 = float(w[idx])
    psi = V[:, idx]
    mask = np.arange(len(w)) != idx
    denom = k0 - w
    gap = float(np.min(np.abs(denom[mask]))) if np.any(mask) else float('inf')
    if np.any(mask) and (not np.isfinite(gap) or gap <= 8*np.finfo(float).eps*max(1.0, float(np.linalg.norm(K0,2)))):
        raise SystemExit('selected eigenvalue is not separated at working precision')

    inv = np.zeros_like(denom, float)
    inv[mask] = 1.0 / denom[mask]
    S = (V * inv) @ V.conj().T
    P = np.outer(psi, psi.conj())

    epsilon_cell = float(np.real(np.trace(P @ K1v)) / (4.0 * Ncells))
    mixed = np.trace(P @ K2) + np.trace(P @ K1q @ S @ K1m) + np.trace(P @ K1m @ S @ K1q)
    chi_cell = float(np.real(mixed) / Ncells)
    epsilon_R = ZR * epsilon_cell
    chi_hat = ZR * chi_cell

    out = {
        'classification': 'FINITE_REGULATOR_SPECTRAL_RESPONSE',
        'selected_index': idx,
        'kappa0_local_branch': k0,
        'N_cells': Ncells,
        'qell': qell,
        'continuum_density_factor': ZR,
        'epsilon0_src_R_homogeneous_frame_response': epsilon_R,
        'chi_hat_TT_R_q': chi_hat,
        'raw_cell_responses': {
            'epsilon_cell': epsilon_cell,
            'chi_TT_cell_q': chi_cell,
        },
        'spectral_gap': gap if np.isfinite(gap) else None,
        'source_basis': source_basis,
        'fourier_adjoint_error': adjoint_error if source_basis == 'fourier_pair' else None,
        'validation_scope': 'Floating-point finite-matrix evaluation; continuum convergence and rigorous input error bounds are separate requirements.',
        'hermiticity_errors': {name: hermitian_error(A) for name, A in mats.items()},
        'definitions': {
            'continuum_density_factor': '(tau_star/delta_t_R)*(ell_star^3/v_cell_R)',
            'epsilon0_src_R': 'Z_R*Re Tr(P0 K1_vol)/(4 N_cells)',
            'chi_hat_TT_R_q': 'Z_R*Re Tr[P0(K2_qmq+K1_q S0 K1_mq+K1_mq S0 K1_q)]/N_cells'
        },
        'physical_limit': (
            'epsilon0_src is the common regulator/thermodynamic limit of epsilon0_src_R for the specified source lift. '
            'The static tensor stiffness is four times the q^2 coefficient of the q=0-subtracted chi_hat_TT response. '
            'Canonical source normalization and the Lorentzian dynamical-metric certificate map these source coordinates '
            'to the covariant epsilon_0 and gamma_R coefficients.'
        ),
        'comparison_policy': 'Experimental G and Lambda are reserved for terminal comparison after forward QICT extraction.'
    }
    txt = json.dumps(out, indent=2, sort_keys=True) + '\n'
    if a.output:
        Path(a.output).write_text(txt)
    print(txt, end='')


if __name__ == '__main__':
    main()
