#!/usr/bin/env python3
"""Check finite-moment exterior Schur self-energy bounds without experimental inputs.

A benchmark on small unitary matrices is a numerical consistency test, not a
calculation of physical charged-lepton, quark or neutrino pole self-energies.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def unitary_model(seed=12017, dimension=9):
    gen = np.random.default_rng(seed)
    a = gen.normal(size=(dimension, dimension)) + 1j * gen.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(a)
    return q * np.conjugate(np.diag(r) / np.abs(np.diag(r)))


def reconstruct(u, v, z, n):
    d = v.shape[1]
    unit = np.eye(d, dtype=complex)
    c = []
    p = np.eye(u.shape[0], dtype=complex)
    for _ in range(n + 1):
        c.append(v.conj().T @ p @ v)
        p = p @ u
    k = c[1]
    leakage = np.linalg.norm(unit - k.conj().T @ k, ord=2)
    r = abs(z)
    eps = 1 / (r ** (n + 1) * (r - 1))
    m = r + 1 + leakage / (r - 1)
    assert m * eps < 1
    gn = sum(c[j] / z ** (j + 1) for j in range(n + 1))
    g = v.conj().T @ np.linalg.solve(z * np.eye(u.shape[0]) - u, v)
    sig = z * unit - k - np.linalg.inv(g)
    sign = z * unit - k - np.linalg.inv(gn)
    bound = m * m * eps / (1 - m * eps)
    err = np.linalg.norm(sig - sign, ord=2)
    assert err <= bound * (1 + 2e-11)
    assert np.linalg.norm(sig, ord=2) <= leakage / (r - 1) + 1e-12
    return dict(depth=n, radius=r, leakage_square=float(leakage),
                exact_norm=float(np.linalg.norm(sig, 2)),
                approximant_norm=float(np.linalg.norm(sign, 2)),
                actual_error=float(err), analytical_upper_evaluated_float64=float(bound),
                moments_norm_error_bound=float(eps))


def reconstruct_inexact(u, v, z, n, seed=31019):
    """Validate the proven a posteriori formula on float64 test contractions."""
    gen = np.random.default_rng(seed + n)
    d = v.shape[1]
    pwr = np.eye(u.shape[0], dtype=complex)
    moments = []
    for j in range(n + 1):
        moments.append(v.conj().T @ pwr @ v)
        pwr = pwr @ u
    eta = {j: 1.0e-5 * (1 + j / n) for j in range(1, n + 1)}
    measured = [moments[0]]
    for j in range(1, n + 1):
        e = gen.normal(size=(d, d)) + 1j * gen.normal(size=(d, d))
        e *= (0.7 * eta[j] / np.linalg.norm(e, 2))
        measured.append(moments[j] + e)
        assert np.linalg.norm(measured[j] - moments[j], 2) <= eta[j] * (1+1e-9)
    khat = measured[1]
    r = abs(z)
    l2 = min(1.0, np.linalg.norm(np.eye(d) - khat.conj().T @ khat, 2)
             + (2+eta[1])*eta[1])
    en = 1.0/(r**(n+1)*(r-1)) + sum(eta[j]/r**(j+1) for j in range(1,n+1))
    ml = r+1+l2/(r-1)
    assert ml*en < 1
    ghat = sum(measured[j]/z**(j+1) for j in range(n+1))
    g = v.conj().T @ np.linalg.solve(z * np.eye(u.shape[0]) - u, v)
    sigma = z*np.eye(d) - moments[1] - np.linalg.inv(g)
    sigma_hat = z*np.eye(d) - khat - np.linalg.inv(ghat)
    err = np.linalg.norm(sigma-sigma_hat, 2)
    bound = eta[1] + ml**2*en/(1-ml*en)
    assert err <= bound*(1+1e-9)
    assert np.linalg.norm(sigma,2) <= l2/(r-1)+1e-11
    return dict(depth=n, moment_error_budget=float(en),
                observed_float64_error=float(err),
                analytical_upper_evaluated_float64=float(bound),
                bound_satisfied=bool(err <= bound*(1+1e-9)))


def exact_cyclic_witness():
    from fractions import Fraction
    n, z = 4, 2
    out = {}
    for p in (-1, 1):
        # U_p acts as a cyclic shift, with last link equal to p.
        u = np.zeros((n + 1, n + 1), dtype=complex)
        for j in range(n):
            u[j + 1, j] = 1
        u[0, n] = p
        v = np.zeros((n + 1, 1), dtype=complex)
        v[0, 0] = 1
        assert all(abs((v.conj().T @ np.linalg.matrix_power(u, k) @ v)[0, 0]) < 1e-14 for k in range(1, n + 1))
        g = (v.conj().T @ np.linalg.inv(z*np.eye(n+1) - u) @ v)[0,0]
        sg = z - 1/g
        exact_sigma = Fraction(p, z ** n)
        assert abs(sg - float(exact_sigma)) < 1e-13
        out[str(p)] = {'sigma_at_two_exact': str(exact_sigma),
                       'first_four_moments': [1,0,0,0,0]}
    return out


def inspect_physical_inputs(root):
    rel = ('reproducibility/results/physical_g2_execution/'
           'PHYSICAL_G2_CONTRACT_AUDIT.json')
    p = Path(root) / rel
    if not p.is_file():
        return {'status':'physical_payload_not_provided', 'beta_f':None}
    payload = json.loads(p.read_text())
    miss = list(payload.get('missing_physical_objects', []))
    return {'status': 'physical_moments_available' if not miss else 'full_physical_moments_not_available',
            'missing_physical_objects': miss,
            'physical_beta_e': None,
            'physical_beta_mu': None,
            'physical_beta_tau': None,
            'physical_beta_quark': None,
            'physical_beta_neutrino': None,
            'interpretation':'No physical self-energy numbers are inferred from an incomplete physical operator or from experimental mass ratios.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--out',default=None)
    ns=ap.parse_args()
    u=unitary_model()
    v=np.eye(u.shape[0],dtype=complex)[:,:6]
    z=2+0.35j
    checks=[reconstruct(u,v,z,n) for n in (4,8,12)]
    assert checks[-1]['actual_error'] < checks[0]['actual_error']
    result={
        'classification':'MATHEMATICAL_EXTERIOR_RESOLVENT_CHECK_NOT_PHYSICAL_POLE_MASSES',
        'theorem':'Supplementary s119 finite-moment self-energy bound',
        'small_matrix_checks':checks,
        'inexact_moment_checks':[reconstruct_inexact(u,v,z,n) for n in (4,8,12)],
        'exact_finite_moment_nonidentifiability_witness':exact_cyclic_witness(),
        'QICT_physical_input_status':inspect_physical_inputs(ns.root),
        'uses_experimental_measurements':False,
        'numerical_method':'deterministic complex QR benchmark (float64 consistency test); exact cyclic witness checked against rational fractions',
    }
    output=Path(ns.out) if ns.out else Path(ns.root)/'reproducibility/results/EXTERIOR_SELFENERGY_MOMENT_VALIDATION.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'mathematical_check':'PASS', 'depths':[c['depth'] for c in checks],
                      'physical_beta_values':'not computable from current operator payload'},indent=2))


if __name__=='__main__':
    main()
