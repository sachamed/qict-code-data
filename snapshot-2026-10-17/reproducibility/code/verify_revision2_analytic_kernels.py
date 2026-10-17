#!/usr/bin/env python3
"""Independent finite checks for the analytic corrections in revision 2.

Run from any directory. Exact SymPy checks verify the analytic-pole and
self-adjoint Schur examples. NumPy checks exercise unitary compression,
central-power distinctions and the complete Weyl basis. These tests do not
claim a physical spectrum or continuum convergence.
"""
from pathlib import Path
import json
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproducibility/results/revision2_analytic_checks'

def norm(a):
    return float(np.linalg.norm(a, 2))

def commutant_dimension(*generators):
    n = generators[0].shape[0]
    equations = np.vstack([np.kron(g.T, np.eye(n))-np.kron(np.eye(n), g)
                           for g in generators])
    s = np.linalg.svd(equations, compute_uv=False)
    return int(n*n-np.count_nonzero(s > 1e-10))

def main():
    rng = np.random.default_rng(20261016)
    w, _ = np.linalg.qr(rng.normal(size=(12, 12))+1j*rng.normal(size=(12, 12)))
    v, _ = np.linalg.qr(rng.normal(size=(12, 6))+1j*rng.normal(size=(12, 6)))
    def green(z):
        return v.conj().T @ np.linalg.solve(z*np.eye(12)-w, v)
    errors = []
    for z in [1.2+0.4j, 2.0, 0.3-0.2j, -1.8j]:
        errors.append(norm(green(1/np.conj(z)).conj().T - z*np.eye(6)+z*z*green(z)))
    assert max(errors) < 1e-12
    w2 = np.array([[0, 1], [-1, 0]], dtype=complex)
    gamma = 2*np.eye(2)-w2
    assert gamma[1, 0] != np.conj(gamma[0, 1])
    z = sp.symbols('z')
    a = sp.Matrix([[z, 1], [z, 0]])
    u, vleft = sp.Matrix([1, 0]), sp.Matrix([0, 1])
    d = (vleft.T*a.diff(z)*u)[0]
    residue = (z*a.inv()).applyfunc(lambda x: sp.limit(x, z, 0))
    assert (vleft.T*u)[0] == 0 and d == 1
    assert residue == u*vleft.T/d
    root = 2-sp.sqrt(2)
    zres = sp.simplify(1/(1+1/(root-3)**2))
    assert sp.simplify(zres*root-sp.Rational(1, 2)) == 0
    assert root != sp.Rational(1, 2)
    omega = np.exp(2j*np.pi/3)
    x = np.roll(np.eye(3, dtype=complex), 1, axis=0)
    z3 = np.diag([1, omega, omega**2])
    assert norm(z3@x-omega*x@z3) < 1e-14
    normalized = np.kron(np.eye(2), x)
    separated = np.kron(np.diag([1, np.exp(1j*np.pi/6)]), x)
    clocks = np.kron(np.eye(2), z3)
    d_normalized = commutant_dimension(normalized, clocks)
    d_separated = commutant_dimension(separated, clocks)
    assert (d_normalized, d_separated) == (4, 2)
    y = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3))
    basis = [np.linalg.matrix_power(x, a)@np.linalg.matrix_power(z3, b)
             for a in range(3) for b in range(3)]
    rebuilt = sum(np.trace(b.conj().T@y)/3*b for b in basis)
    weyl_error = norm(rebuilt-y)
    assert weyl_error < 1e-13
    phases = [sp.Rational(3, 2)*dt for dt in [sp.Rational(1), sp.Rational(1, 2), sp.Rational(1, 4)]]
    energies = [2/dt*phase for dt, phase in zip([sp.Rational(1),sp.Rational(1,2),sp.Rational(1,4)],phases)]
    assert energies == [3, 3, 3]
    # Independent quadrature of a correlated lognormal covariance.
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    weights = weights / np.sqrt(np.pi)
    log_cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    means = np.array([0.2, -0.1])
    grid = np.stack(np.meshgrid(nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 2)
    joint_weights = np.outer(weights, weights).ravel()
    samples = np.exp(-(means + np.sqrt(2)*grid@np.linalg.cholesky(log_cov).T))
    quadrature_mean = joint_weights@samples
    centered = samples-quadrature_mean
    quadrature_cov = centered.T@(joint_weights[:,None]*centered)
    exact_mean = np.exp(-means+np.diag(log_cov)/2)
    exact_cov = np.outer(exact_mean,exact_mean)*np.expm1(log_cov)
    covariance_error = norm(quadrature_cov-exact_cov)
    assert covariance_error < 1e-13
    result = {
        'status': 'PASS',
        'scope': 'Independent finite analytic checks; no physical or continuum closure claim.',
        'seed': 20261016,
        'unitary_compressed_reciprocity_max_operator_error': max(errors),
        'nonhermitian_chiral_counterexample': 'EXACT_INTEGER_MATRIX',
        'analytic_residue_orthogonal_null_vectors': 'EXACT_SYMBOLIC_PASS',
        'schur_pole': str(root),
        'residue_times_schur_coefficient': '1/2',
        'weyl_commutant_dimension_common_central_powers': d_normalized,
        'weyl_commutant_dimension_distinct_central_powers': d_separated,
        'full_nine_word_Weyl_reconstruction_error': weyl_error,
        'phase_scale_conversion': 'EXACT_RATIONAL_PASS',
        'lognormal_covariance_quadrature_error': covariance_error
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'ANALYTIC_CHECKS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
