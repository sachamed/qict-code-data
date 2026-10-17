#!/usr/bin/env python3
"""QICT-D21 UV-kernel and reduced charged-lepton closure audit.

The script checks exact algebraic identities used in Supplement AH and
reproduces the reduced connected hierarchy.  Finite-dimensional unitary
matrices are used only as a numerical sanity check of the general compression
lemmas; they are not substituted for the full 3D QICT contraction.

Exit status 0 means that the stated mathematical identities reproduce.  It
does not mean that the microscopic thermodynamic value of K_UV has already
been computed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 80

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "UV_KERNEL_CLOSURE_AUDIT.json"


def relerr(pred, ref):
    return pred / ref - 1


def passcheck(name, condition, detail):
    checks.append({"name": name, "pass": bool(condition), "detail": detail})


# Reduced QICT coefficient.
R_BCC = mp.gamma(mp.mpf(1) / 4) ** 4 / (32 * mp.pi**3)
W12 = mp.mpf("0.8123368752121693025569639180601123743")
alpha_fac = R_BCC * mp.sqrt(W12) * mp.cos(mp.pi / 63)
delta_c3 = -mp.log(1 - alpha_fac**3) / 96
alpha_conn = alpha_fac + delta_c3

P = {"e": mp.mpf(55), "mu": mp.mpf(21), "tau": mp.mpf(3)}
ratios = {
    "mu_over_e": mp.e ** ((P["e"] - P["mu"]) * alpha_conn),
    "tau_over_mu": mp.e ** ((P["mu"] - P["tau"]) * alpha_conn),
    "tau_over_e": mp.e ** ((P["e"] - P["tau"]) * alpha_conn),
}

# PDG 2026 external audit values, MeV.
me = mp.mpf("0.51099895069")
mmu = mp.mpf("105.6583755")
mtau = mp.mpf("1776.93")
pdg = {
    "mu_over_e": mmu / me,
    "tau_over_mu": mtau / mmu,
    "tau_over_e": mtau / me,
}
residuals = {k: relerr(ratios[k], pdg[k]) for k in ratios}

# Historical Planck-flow target.  This is an inverse diagnostic, not a derivation.
Mpl_GeV = mp.mpf("1.220890e19")
Lambda_hist_GeV = mp.mpf("2.84400317055")
K_required = alpha_conn * mp.log(Mpl_GeV / Lambda_hist_GeV)

# Generic finite-dimensional check of the compression theorem.
rng = np.random.default_rng(210963)
Z = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
Q, R = np.linalg.qr(Z)
phase = np.diag(R)
phase = phase / np.where(np.abs(phase) > 0, np.abs(phase), 1)
U = Q @ np.diag(np.conj(phase))
Pmat = np.diag([1, 1, 1, 1, 0, 0, 0, 0]).astype(complex)
Qmat = np.eye(8, dtype=complex) - Pmat
T = Pmat @ U @ Pmat
lhs = Pmat - T.conj().T @ T
rhs = Pmat @ U.conj().T @ Qmat @ U @ Pmat
leakage_residual = np.linalg.norm(lhs - rhs, ord=2)
leakage_eigs = np.linalg.eigvalsh((lhs + lhs.conj().T) / 2)

# Restrict to P block and form K=-1/2 log(T^*T) on nonzero support.
Tb = T[:4, :4]
X = Tb.conj().T @ Tb
xeigs = np.linalg.eigvalsh((X + X.conj().T) / 2)
positive = xeigs[xeigs > 1e-14]
Ktrace = float(-0.5 * np.mean(np.log(positive)))

# Gauge/basis conjugation sanity check using a block-diagonal unitary commuting with P.
ph = np.exp(1j * np.array([0.11, -0.37, 0.51, -0.22, 0.4, 0.8, -0.5, 0.3]))
G = np.diag(ph)
Ug = G @ U @ G.conj().T
Tg = Pmat @ Ug @ Pmat
Xg = Tg[:4, :4].conj().T @ Tg[:4, :4]
xgeigs = np.linalg.eigvalsh((Xg + Xg.conj().T) / 2)
positive_g = xgeigs[xgeigs > 1e-14]
Ktrace_g = float(-0.5 * np.mean(np.log(positive_g)))

checks = []
passcheck("01_alpha_fac", abs(alpha_fac - mp.mpf("0.1567662572156974013416910182029180188")) < mp.mpf("1e-35"), str(alpha_fac))
passcheck("02_C3_trace_log", abs(delta_c3 - mp.mpf("0.00004020911098348874431970666359839088915")) < mp.mpf("1e-38"), str(delta_c3))
passcheck("03_alpha_conn", abs(alpha_conn - mp.mpf("0.1568064663266808900860107248665164097")) < mp.mpf("1e-35"), str(alpha_conn))
passcheck("04_cost_difference_mu_e", P["e"] - P["mu"] == 34, str(P["e"] - P["mu"]))
passcheck("05_cost_difference_tau_mu", P["mu"] - P["tau"] == 18, str(P["mu"] - P["tau"]))
passcheck("06_cost_difference_tau_e", P["e"] - P["tau"] == 52, str(P["e"] - P["tau"]))
passcheck("07_ratio_mu_e", abs(ratios["mu_over_e"] - mp.mpf("206.7312943550371")) < mp.mpf("1e-12"), str(ratios["mu_over_e"]))
passcheck("08_ratio_tau_mu", abs(ratios["tau_over_mu"] - mp.mpf("16.81912099859112")) < mp.mpf("1e-13"), str(ratios["tau_over_mu"]))
passcheck("09_ratio_tau_e", abs(ratios["tau_over_e"] - mp.mpf("3477.038653952728")) < mp.mpf("1e-11"), str(ratios["tau_over_e"]))
passcheck("10_not_all_PDG_residuals_below_1e-4", not all(abs(x) < mp.mpf("1e-4") for x in residuals.values()), {k: str(v) for k, v in residuals.items()})
passcheck("11_leakage_identity", leakage_residual < 1e-12, leakage_residual)
passcheck("12_leakage_positive", float(leakage_eigs.min()) > -1e-12, leakage_eigs.tolist())
passcheck("13_compression_spectrum_below_one", float(xeigs.max()) <= 1 + 1e-12, xeigs.tolist())
passcheck("14_uv_kernel_nonnegative", Ktrace >= -1e-12, Ktrace)
passcheck("15_gauge_conjugation_invariant", abs(Ktrace - Ktrace_g) < 1e-12, {"before": Ktrace, "after": Ktrace_g})
passcheck("16_Planck_target_is_inverse_diagnostic", abs(K_required - mp.mpf("6.72754375695026")) < mp.mpf("1e-13"), str(K_required))

assert all(item["pass"] for item in checks)

report = {
    "status": "16/16 PASS",
    "scientific_status": "operator definition and algebraic identities verified; microscopic thermodynamic K_UV not yet evaluated",
    "alpha_factorised": str(alpha_fac),
    "delta_alpha_C3": str(delta_c3),
    "alpha_connected_reduced": str(alpha_conn),
    "reduced_mass_ratios": {k: str(v) for k, v in ratios.items()},
    "PDG_2026_relative_residuals": {k: str(v) for k, v in residuals.items()},
    "historical_K_UV_inverse_target": str(K_required),
    "checks": checks,
}
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
