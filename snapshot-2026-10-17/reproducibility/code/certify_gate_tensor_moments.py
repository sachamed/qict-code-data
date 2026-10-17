#!/usr/bin/env python3
"""Exact first-principle finite-step sourced-unitary tensor moment checks.

This certificate checks algebraic identities of source-deformed unitary gates,
NOT the physical QICT interacting C10/C11 tensor spectrum or a graviton.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'GATE_DERIVED_TENSOR_MOMENTS.json'
RNG = np.random.default_rng(20261015)


def random_hermitian(n):
    z = RNG.standard_normal((n, n)) + 1j * RNG.standard_normal((n, n))
    return (z + z.conj().T) / (2 * np.sqrt(n))


def evolve(base, insertions, h):
    ident = np.eye(base[0].shape[0], dtype=complex)
    out = ident
    for j, g in enumerate(base):
        t = h[0] * insertions[j][0] + h[1] * insertions[j][1]
        out = (expm(-1j*t) @ g) @ out
    return out


def source_derivative(base, insertions):
    n = base[0].shape[0]
    prefix = np.eye(n, dtype=complex)
    accum = [np.zeros((n, n), complex), np.zeros((n, n), complex)]
    for j, g in enumerate(base):
        for a in range(2):
            tj = g.conj().T @ insertions[j][a] @ g
            accum[a] += prefix.conj().T @ tj @ prefix
        prefix = g @ prefix
    return accum


def source_measure(u, deriv, eps):
    ev, eigvec = np.linalg.eig(u)
    ev = ev / np.abs(ev)
    select = np.argmin(np.abs(np.angle(ev)))
    psi = eigvec[:, select]
    psi /= np.linalg.norm(psi)
    w = u / ev[select]
    eta = np.column_stack([(t - np.vdot(psi, t @ psi) * np.eye(len(psi))) @ psi
                           for t in deriv])
    gram = eta.conj().T @ eta
    m1 = eta.conj().T @ w @ eta
    defect = 2*gram - m1 - m1.conj().T
    # Eigenvector matrix is unitary up to roundoff for the finite normal matrix.
    ww, basis = np.linalg.eig(w)
    projections = basis.conj().T @ eta
    angles = np.angle(ww)
    close = np.abs(angles) <= eps + 1e-13
    low = projections[close].conj().T @ projections[close]
    total = projections.conj().T @ projections
    d = 2*np.sin(eps/2)
    margin = low - (gram - defect / d**2)
    return {
        'hermitian_defect': float(np.linalg.norm(defect-defect.conj().T, 2)),
        'toeplitz_lowest': float(min(np.linalg.eigvalsh(np.block([
            [eta.conj().T @ np.linalg.matrix_power(w, k-j) @ eta for k in range(5)]
            for j in range(5)])))),
        'source_weight_min': float(min(np.linalg.eigvalsh(low))),
        'lower_bound_margin': float(min(np.linalg.eigvalsh((margin+margin.conj().T)/2))),
        'high_weight_max': float(max(np.linalg.eigvalsh(gram-low))),
        'moment_sum_error': float(np.linalg.norm(total-gram, 2)),
        'defect_positive_min': float(min(np.linalg.eigvalsh(defect))),
    }


def main():
    trials=64
    largest_derivative_error=0.0
    min_low_margin=1e10
    min_toeplitz=-1e10
    largest_sum_error=0.0
    n=6
    for _ in range(trials):
        base=[expm(-1j*random_hermitian(n)) for __ in range(4)]
        insertions=[(random_hermitian(n),random_hermitian(n)) for __ in base]
        u=evolve(base,insertions,(0.,0.))
        deriv=source_derivative(base,insertions)
        step=2e-6
        for a in range(2):
            h=np.zeros(2);h[a]=step
            plus=evolve(base,insertions,h)
            minus=evolve(base,insertions,-h)
            fd=1j*u.conj().T@(plus-minus)/(2*step)
            largest_derivative_error=max(largest_derivative_error,float(np.linalg.norm(fd-deriv[a],2)))
            assert np.linalg.norm(deriv[a]-deriv[a].conj().T,2)<2e-12
        c=source_measure(u,deriv,0.65)
        min_low_margin=min(min_low_margin,c['lower_bound_margin'])
        min_toeplitz=min(min_toeplitz,c['toeplitz_lowest']) if _ else c['toeplitz_lowest']
        largest_sum_error=max(largest_sum_error,c['moment_sum_error'])
        assert c['lower_bound_margin']>=-2e-10
        assert c['toeplitz_lowest']>=-2e-10
        assert c['moment_sum_error']<2e-10
        assert c['defect_positive_min']>=-2e-10

    # Positive low-phase weight with an explicit sourced-gate toy system.
    U=np.diag(np.exp(-1j*np.array([0.,.025,.045,1.2,1.6])))
    t0=np.zeros((5,5),complex); t0[1,0]=t0[0,1]=1
    t1=np.zeros((5,5),complex); t1[2,0]=t1[0,2]=1
    test=source_measure(U,[U.conj().T@t0@U,U.conj().T@t1@U],0.1)
    assert test['source_weight_min']>0.99 and test['lower_bound_margin']>-1e-10

    # Distinct local positive generators with precisely equal sampled evolution.
    delta_t=0.37
    h0=np.zeros((2,2))
    h1=np.diag([0,2*np.pi/delta_t])
    alias_error=float(np.linalg.norm(expm(-1j*delta_t*h0)-expm(-1j*delta_t*h1),2))
    assert alias_error<5e-15
    assert largest_derivative_error < 2e-8
    output={
      'classification':'EXACT_SOURCED_GATE_IDENTITIES_NOT_A_PHYSICAL_QICT_GRAVITON',
      'trial_seed':20261015,'complex_noncommuting_gate_trials':trials,
      'gates_per_trial':4,'hilbert_dimension':6,
      'max_product_derivative_residual':largest_derivative_error,
      'min_toeplitz_eigenvalue':min_toeplitz,
      'min_low_weight_bound_margin':min_low_margin,
      'max_total_spectral_weight_residual':largest_sum_error,
      'two_source_positive_overlap_toy':test,
      'onsite_energy_alias_unitary_error':alias_error,
      'energy_alias_excited_gap':2*np.pi/delta_t,
      'physical_QICT_interacting_tensor_operator_available':False,
      'positive_energy_branch_derived_from_microscopic_regulator':False,
      'spin_two_graviton_massless_pole_established':False,
      'tests_passed':True,
    }
    OUT.write_text(json.dumps(output, indent=2,sort_keys=True)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
