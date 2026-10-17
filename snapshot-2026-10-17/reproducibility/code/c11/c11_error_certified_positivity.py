#!/usr/bin/env python3
"""Rigorous C11 block-Toeplitz positivity acceptance.

Two logically distinct quantities are required:

1. certified moment errors ||C_k-C_hat_k||_F <= eps_k;
2. a certified lower bound lambda_hat_lb <= lambda_min(T_hat_N) for the
   *computed* block-Toeplitz matrix itself (for example from interval LDL^*,
   exact rational arithmetic, or another validated linear-algebra certificate).

Agreement between two floating-point reconstructions is retained only as a
regression diagnostic and is never converted into eps_k.

With certified eps_k,

 ||Delta T_N||_2 <= ||Delta T_N||_F
 <= sqrt((N+1) eps_0^2 + 2 sum_{k=1}^N (N+1-k) eps_k^2),

and therefore

 lambda_min(T_N) >= lambda_hat_lb - ||Delta T_N||_F.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def block_toeplitz(C, N):
    C = np.asarray(C, complex)
    d = C.shape[1]
    T = np.zeros((d * (N + 1), d * (N + 1)), complex)
    for i in range(N + 1):
        for j in range(N + 1):
            k = j - i
            T[d*i:d*(i+1), d*j:d*(j+1)] = C[k] if k >= 0 else C[-k].conj().T
    return (T + T.conj().T) / 2


def perturbation_bound(eps, N):
    eps = np.asarray(eps, float)
    if len(eps) <= N:
        raise ValueError('need one certified error bound per moment through N')
    if np.any(~np.isfinite(eps)) or np.any(eps < 0):
        raise ValueError('moment error bounds must be finite and nonnegative')
    sq = (N + 1) * eps[0]**2 + 2 * sum((N + 1 - k) * eps[k]**2 for k in range(1, N + 1))
    return float(np.sqrt(sq))


def certify(C, eps, lambda_hat_lower_bound, N=None):
    C = np.asarray(C, complex)
    if C.ndim != 3 or C.shape[1:] != (6, 6):
        raise ValueError('moments must be (M,6,6)')
    if N is None:
        N = min(16, len(C) - 1)
    T = block_toeplitz(C, N)
    numerical_lambda = float(np.linalg.eigvalsh(T).min())
    pert = perturbation_bound(eps, N)
    exact_lower = float(lambda_hat_lower_bound - pert)
    return {
        'status': 'PASS_ERROR_CERTIFIED_C11_POSITIVITY' if exact_lower >= 0 else 'C11_POSITIVITY_NOT_CERTIFIED',
        'depth': N,
        'numerical_lambda_min_diagnostic': numerical_lambda,
        'certified_computed_toeplitz_lambda_min_lower_bound': float(lambda_hat_lower_bound),
        'toeplitz_perturbation_fro_upper_bound': pert,
        'certified_exact_lambda_min_lower_bound': exact_lower,
        'positive_certified': bool(exact_lower >= 0),
        'exact_structural_positivity':{'status':'PASS_EXACT_UNITARY_MOMENT_TOEPLITZ_POSITIVITY_THEOREM','statement':'For exact C_k=V^*U^kV with U unitary, T_N is a Gram matrix and is positive semidefinite.'},
    }


def load_certificate(path: Path, moments_path: Path, depth: int):
    obj = json.loads(path.read_text())
    if obj.get('status') != 'PASS_RIGOROUS_MOMENT_AND_TOEPLITZ_ERROR_BOUNDS':
        raise ValueError('error-bound certificate status is not PASS_RIGOROUS_MOMENT_AND_TOEPLITZ_ERROR_BOUNDS')
    if obj.get('moments_sha256') != sha256(moments_path):
        raise ValueError('error-bound certificate does not match moments file digest')
    if int(obj.get('depth', -1)) != int(depth):
        raise ValueError('error-bound certificate depth mismatch')
    eps = np.asarray(obj.get('moment_fro_error_bounds', []), float)
    lam_lb = obj.get('computed_toeplitz_lambda_min_lower_bound')
    if lam_lb is None:
        raise ValueError('certificate lacks computed_toeplitz_lambda_min_lower_bound')
    return eps, float(lam_lb), obj


def diagnostic_without_certificate(C, z, N, moments_path):
    T = block_toeplitz(C, N)
    out = {
        'status': 'RIGOROUS_MOMENT_AND_TOEPLITZ_ERROR_CERTIFICATION_STAGE',
        'depth': N,
        'positive_certified': False,
        'numerical_lambda_min_diagnostic': float(np.linalg.eigvalsh(T).min()),
        'moments_sha256': sha256(moments_path),
        'acceptance_rule': 'Floating-point cross-agreement is a regression diagnostic only. A quantitative numerical positivity margin requires independently certified moment-error bounds and a validated lower bound for lambda_min(T_hat_N).',
        'exact_structural_positivity':{'status':'PASS_EXACT_UNITARY_MOMENT_TOEPLITZ_POSITIVITY_THEOREM','statement':'For exact C_k=V^*U^kV with U unitary, T_N is a Gram matrix and is positive semidefinite.'},
    }
    if 'spectral_moments' in z.files:
        D = C - np.asarray(z['spectral_moments'], complex)
        out['direct_vs_spectral_max_fro_discrepancy_diagnostic'] = float(max(np.linalg.norm(x) for x in D))
        out['direct_vs_spectral_max_entrywise_discrepancy_diagnostic'] = float(np.max(np.abs(D)))
    return out


def self_test():
    # Exact theorem test: C0=I_6 and Ck=0 for k>=1 gives T_N=I exactly.
    # Use exact zero moment errors and the analytical lower bound lambda_min=1.
    N = 8
    C = np.zeros((N + 1, 6, 6), complex)
    C[0] = np.eye(6)
    eps = np.zeros(N + 1)
    out = certify(C, eps, 1.0, N)
    out['status'] = 'PASS_C11_RIGOROUS_ERROR_PROPAGATION_SELF_TEST' if out['positive_certified'] and abs(out['certified_exact_lambda_min_lower_bound'] - 1.0) < 1e-15 else 'FAIL_C11_RIGOROUS_ERROR_PROPAGATION_SELF_TEST'
    out['self_test_basis'] = 'Analytical sequence C0=I6, Ck=0 (k>=1), so T_N=I exactly.'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--moments', type=Path)
    ap.add_argument('--error-certificate', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--depth', type=int)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.self_test:
        out = self_test()
        a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
        print(json.dumps(out, indent=2))
        return

    if a.moments is None:
        raise SystemExit('--moments is required outside --self-test')
    z = np.load(a.moments, allow_pickle=False)
    C = z['direct_moments'] if 'direct_moments' in z.files else z['moments']
    N = a.depth if a.depth is not None else min(16, len(C) - 1)

    if a.error_certificate is None:
        out = diagnostic_without_certificate(C, z, N, a.moments)
    else:
        eps, lam_lb, cert_meta = load_certificate(a.error_certificate, a.moments, N)
        out = certify(C, eps, lam_lb, N)
        out['moments_sha256'] = sha256(a.moments)
        out['error_certificate_sha256'] = sha256(a.error_certificate)
        out['error_bound_method'] = cert_meta.get('method')
        if 'spectral_moments' in z.files:
            D = C - np.asarray(z['spectral_moments'], complex)
            out['direct_vs_spectral_max_fro_discrepancy_diagnostic'] = float(max(np.linalg.norm(x) for x in D))
            out['cross_agreement_used_as_error_bound'] = False

    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
