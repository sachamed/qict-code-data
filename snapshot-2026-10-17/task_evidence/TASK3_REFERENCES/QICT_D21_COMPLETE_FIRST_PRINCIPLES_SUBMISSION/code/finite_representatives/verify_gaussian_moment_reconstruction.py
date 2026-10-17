#!/usr/bin/env python3
"""Validate the QICT unit-circle moment reconstruction on the solved Gaussian C7 spectrum.

This is a benchmark of the reconstruction pipeline only. It is not an interacting calculation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "certificates" / "spectral_moment_reconstruction_certificate.json"

THETA = {
    "u": [0.12283835613997052, 0.5851931160827725, 0.9694083121262547],
    "d": [0.01550012491171076, 0.6149705520485497, 0.9555708985679071],
    "e": [0.17401963748403787, 0.5574263894516671, 0.9804870890570604],
    "nu_D": [0.29471803984885053, 0.46813116464230653, 1.0043581154829457],
}


def newton_polynomial_from_power_sums(power_sums):
    """Return monic polynomial coefficients from p_k=sum_j r_j^k, k=1..d."""
    d = len(power_sums)
    e = np.zeros(d + 1, dtype=np.complex128)
    e[0] = 1.0
    for k in range(1, d + 1):
        acc = 0j
        for i in range(1, k + 1):
            acc += ((-1) ** (i - 1)) * e[k - i] * power_sums[i - 1]
        e[k] = acc / k
    coeff = np.empty(d + 1, dtype=np.complex128)
    coeff[0] = 1.0
    for k in range(1, d + 1):
        coeff[k] = ((-1) ** k) * e[k]
    return coeff


def phase_set(roots):
    phases = np.angle(roots)
    pos = sorted([abs(float(x)) for x in phases])
    # roots occur in +/- pairs; average each adjacent pair after sorting abs values.
    return np.array([(pos[2*i] + pos[2*i+1]) / 2.0 for i in range(3)])


def toeplitz_from_moments(roots, N):
    # equal positive weights; C_n = average r^n
    def c(n):
        if n >= 0:
            return np.mean(roots ** n)
        return np.conjugate(c(-n))
    T = np.empty((N + 1, N + 1), dtype=np.complex128)
    for i in range(N + 1):
        for j in range(N + 1):
            T[i, j] = c(j - i)
    return T


def main():
    results = {}
    worst_phase = 0.0
    worst_radius = 0.0
    worst_poly = 0.0
    min_toeplitz = 1.0
    for sector, th in THETA.items():
        th = np.array(th, dtype=float)
        roots = np.concatenate([np.exp(1j * th), np.exp(-1j * th)])
        p = np.array([np.sum(roots ** k) for k in range(1, 7)], dtype=np.complex128)
        coeff = newton_polynomial_from_power_sums(p)
        rec = np.roots(coeff)
        rec_th = phase_set(rec)
        target = np.sort(th)
        phase_err = float(np.max(np.abs(rec_th - target)))
        radius_err = float(np.max(np.abs(np.abs(rec) - 1.0)))
        # Verify polynomial annihilates exact roots.
        poly_err = float(np.max(np.abs(np.polyval(coeff, roots))))
        T = toeplitz_from_moments(roots, 6)
        eig = np.linalg.eigvalsh((T + T.conj().T) / 2)
        mineig = float(np.min(eig))
        worst_phase = max(worst_phase, phase_err)
        worst_radius = max(worst_radius, radius_err)
        worst_poly = max(worst_poly, poly_err)
        min_toeplitz = min(min_toeplitz, mineig)
        results[sector] = {
            "target_positive_phases": target.tolist(),
            "reconstructed_positive_phases": rec_th.tolist(),
            "max_phase_error": phase_err,
            "max_unit_circle_radius_error": radius_err,
            "exact_root_polynomial_residual": poly_err,
            "toeplitz_min_eigenvalue_N6": mineig,
            "power_sums_n1_to_n6": [[float(z.real), float(z.imag)] for z in p],
        }
    cert = {
        "theory": "QICT C7 publication model",
        "benchmark": "Gaussian C7 unit-circle spectral moment reconstruction",
        "external_data_opened": False,
        "method": "Newton identities from six power moments for six Gaussian roots exp(+-i theta_a)",
        "sectors": results,
        "worst_max_phase_error": worst_phase,
        "worst_unit_circle_radius_error": worst_radius,
        "worst_exact_root_polynomial_residual": worst_poly,
        "minimum_toeplitz_eigenvalue_N6": min_toeplitz,
        "interpretation": "Validation of the moment-to-spectrum pipeline only; no interacting pole is claimed.",
        "status": "PASS" if worst_phase < 1e-10 and worst_radius < 1e-10 and worst_poly < 1e-10 and min_toeplitz > -1e-12 else "FAIL",
    }
    OUT.write_text(json.dumps(cert, indent=2) + "\n")
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
