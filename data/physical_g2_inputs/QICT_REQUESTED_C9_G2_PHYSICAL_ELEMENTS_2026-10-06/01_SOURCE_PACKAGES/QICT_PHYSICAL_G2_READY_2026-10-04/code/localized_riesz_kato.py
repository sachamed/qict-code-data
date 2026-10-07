#!/usr/bin/env python3
"""Localized finite-dimensional Riesz--Kato projector primitive.

Certification rule
------------------
A sampled minimum of sigma_min(zI-A) is *not* a lower bound on an entire
contour.  For the circular contour z(theta)=c+r exp(i theta), singular-value
Lipschitz continuity gives

  |sigma_min(z(theta)I-A)-sigma_min(z(phi)I-A)| <= |z(theta)-z(phi)|.

For N equally spaced midpoint nodes, every contour point lies at chord distance
at most 2 r sin(pi/(2N)) from a node.  Hence, if ell_j are independently
certified nodewise lower bounds,

  inf_Gamma sigma_min(zI-A) >= min_j ell_j - 2 r sin(pi/(2N)).

The code therefore keeps the raw sampled minimum as a diagnostic only.  A
physical protecting gap is emitted only when certified nodewise lower bounds
are supplied.  This closes the gap between contour sampling and a uniform
resolvent-set statement.
"""
from __future__ import annotations
import argparse, json, hashlib, math
from pathlib import Path
import numpy as np


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_operator(path: Path, key: str = 'operator') -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    if key not in z.files:
        raise ValueError(f'{path} lacks {key}')
    A = np.asarray(z[key], complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('operator must be square')
    return A


def contour_nodes(center: complex, radius: float, nquad: int) -> np.ndarray:
    if radius <= 0 or nquad < 8:
        raise ValueError('radius must be positive and nquad >= 8')
    theta = 2 * np.pi * (np.arange(nquad) + 0.5) / nquad
    return center + radius * np.exp(1j * theta)


def riesz_projector_sampled(A: np.ndarray, center: complex, radius: float, nquad: int = 128):
    """Midpoint contour quadrature plus *diagnostic* node singular values.

    The returned sampled minimum is never treated as a uniform contour bound.
    """
    N = A.shape[0]
    I = np.eye(N, dtype=complex)
    P = np.zeros_like(A)
    svals = []
    nodes = contour_nodes(center, radius, nquad)
    for z in nodes:
        M = z * I - A
        svals.append(float(np.linalg.svd(M, compute_uv=False)[-1]))
        R = np.linalg.solve(M, I)
        # dz/(2pi i) for midpoint rule on a circular contour.
        P += ((z - center) / nquad) * R
    return P, np.asarray(svals, float)


def uniform_contour_lower_bound(node_lower_bounds: np.ndarray, radius: float, nquad: int) -> dict:
    """Convert certified nodewise bounds into a whole-contour lower bound.

    For midpoint nodes the maximum angular displacement is pi/N, hence the
    maximum chord distance is 2 r sin(pi/(2N)).
    """
    ell = np.asarray(node_lower_bounds, float).reshape(-1)
    if len(ell) != nquad:
        raise ValueError(f'expected {nquad} certified node lower bounds, got {len(ell)}')
    if np.any(~np.isfinite(ell)) or np.any(ell <= 0):
        raise ValueError('certified node lower bounds must be finite and positive')
    interpolation_loss = 2.0 * radius * math.sin(math.pi / (2.0 * nquad))
    lower = float(np.min(ell) - interpolation_loss)
    return {
        'certified_node_minimum': float(np.min(ell)),
        'contour_lipschitz_interpolation_loss': float(interpolation_loss),
        'uniform_contour_singular_value_lower_bound': lower,
        'uniform_bound_positive': bool(lower > 0),
        'proof': 'Weyl singular-value Lipschitz bound plus nearest-midpoint chord bound 2 r sin(pi/(2N)).',
    }


def load_node_lower_bounds(path: Path, nquad: int) -> np.ndarray:
    if path.suffix.lower() == '.json':
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            for key in ('node_lower_bounds', 'certified_node_lower_bounds', 'lower_bounds'):
                if key in obj:
                    obj = obj[key]
                    break
        arr = np.asarray(obj, float)
    elif path.suffix.lower() == '.npy':
        arr = np.asarray(np.load(path, allow_pickle=False), float)
    elif path.suffix.lower() == '.npz':
        z = np.load(path, allow_pickle=False)
        key = next((k for k in ('node_lower_bounds', 'certified_node_lower_bounds', 'lower_bounds') if k in z.files), None)
        if key is None:
            raise ValueError('NPZ lacks certified node lower-bound array')
        arr = np.asarray(z[key], float)
    else:
        raise ValueError('node lower bounds must be JSON, NPY, or NPZ')
    arr = arr.reshape(-1)
    if len(arr) != nquad:
        raise ValueError(f'expected {nquad} node lower bounds, got {len(arr)}')
    return arr


def positive_density_from_projector(P: np.ndarray) -> np.ndarray:
    """Construct the serialized approximate density by an explicit Gram factor.

    Q=P P^* is positive semidefinite by construction.  No eigenvalue clipping is
    used, so a negative numerical eigenvalue cannot be silently repaired.  The
    distance from this approximate density to the exact Kato density belongs to
    the independently certified Riesz/operator error budget.
    """
    P=np.asarray(P,complex)
    Q=P@P.conj().T
    tr=float(np.trace(Q).real)
    if tr<=0:
        raise ValueError('Riesz contour selected no positive subspace')
    return Q/tr


def projector_diagnostics(A: np.ndarray, P: np.ndarray, rho: np.ndarray) -> dict:
    herm = float(np.linalg.norm(P - P.conj().T))
    idem = float(np.linalg.norm(P @ P - P))
    comm = float(np.linalg.norm(A @ P - P @ A))
    evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return {
        'projector_hermiticity_fro_diagnostic': herm,
        'projector_idempotency_fro_diagnostic': idem,
        'projector_commutator_fro_diagnostic': comm,
        'rho_trace': float(np.trace(rho).real),
        'rho_lambda_min': float(evals.min()),
        'rho_lambda_max': float(evals.max()),
    }


def synthetic_self_test() -> dict:
    # Exactly transparent normal test: eigenvalue 1 has multiplicity two and
    # the circle |z-1|=0.1 encloses precisely that two-dimensional cluster.
    # At every contour node the distance to either enclosed eigenvalue is
    # exactly r; all other listed eigenvalues are farther from the circle.
    A = np.diag(np.array([1, 1, -1, 1j, -1j, np.exp(1j * np.pi / 3)], complex))
    center = 1.0 + 0.0j
    radius = 0.1
    nquad = 256
    P, sampled = riesz_projector_sampled(A, center, radius, nquad)
    # For this constructed test the nodewise lower bound r is analytical.
    node_lb = np.full(nquad, radius, dtype=float)
    whole = uniform_contour_lower_bound(node_lb, radius, nquad)
    rho = positive_density_from_projector(P)
    diag = projector_diagnostics(A, P, rho)
    selected_rank = int(round(np.trace((P + P.conj().T) / 2).real))
    ok = (
        selected_rank == 2
        and diag['projector_idempotency_fro_diagnostic'] < 1e-10
        and whole['uniform_contour_singular_value_lower_bound'] > 0
        and float(np.min(sampled)) >= radius - 1e-12
    )
    return {
        'status': 'PASS_LOCALIZED_RIESZ_KATO_UNIFORM_CONTOUR_SELF_TEST' if ok else 'FAIL_LOCALIZED_RIESZ_KATO_UNIFORM_CONTOUR_SELF_TEST',
        'selected_rank': selected_rank,
        'contour_center': [center.real, center.imag],
        'contour_radius': radius,
        'quadrature_points': nquad,
        'sampled_minimum_contour_singular_value_diagnostic': float(np.min(sampled)),
        'node_bound_source': 'analytic normal-matrix test: two eigenvalues at contour center, nodewise distance exactly r',
        **whole,
        **diag,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--operator', type=Path)
    ap.add_argument('--key', default='operator')
    ap.add_argument('--center-real', type=float)
    ap.add_argument('--center-imag', type=float)
    ap.add_argument('--radius', type=float)
    ap.add_argument('--nquad', type=int, default=128)
    ap.add_argument('--certified-node-lower-bounds', type=Path,
                    help='JSON/NPY/NPZ array of independently certified lower bounds for sigma_min at every contour node')
    ap.add_argument('--out-density', type=Path)
    ap.add_argument('--out-json', type=Path, required=True)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    if a.self_test:
        out = synthetic_self_test()
        a.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
        print(json.dumps(out, indent=2))
        return

    if not a.operator or a.radius is None or a.center_real is None or a.center_imag is None or not a.out_density:
        raise SystemExit('physical mode requires --operator, center, radius and --out-density')

    A = load_operator(a.operator, a.key)
    center = complex(a.center_real, a.center_imag)
    P, sampled = riesz_projector_sampled(A, center, a.radius, a.nquad)
    rho = positive_density_from_projector(P)
    diag = projector_diagnostics(A, P, rho)
    base = {
        'operator_sha256': sha256(a.operator),
        'matrix_dimension': A.shape[0],
        'contour_center': [center.real, center.imag],
        'contour_radius': a.radius,
        'quadrature_points': a.nquad,
        'sampled_minimum_contour_singular_value_diagnostic': float(np.min(sampled)),
        **diag,
    }

    if a.certified_node_lower_bounds is None:
        out = {
            **base,
            'status': 'AWAITING_CERTIFIED_CONTOUR_NODE_LOWER_BOUNDS',
            'protecting_gap_certified': False,
            'uniform_contour_singular_value_lower_bound': None,
            'acceptance': 'A sampled minimum is diagnostic only; physical Kato acceptance requires certified nodewise lower bounds and the whole-contour Lipschitz deduction.',
        }
        a.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
        print(json.dumps(out, indent=2))
        return

    node_lb = load_node_lower_bounds(a.certified_node_lower_bounds, a.nquad)
    whole = uniform_contour_lower_bound(node_lb, a.radius, a.nquad)
    ok = whole['uniform_bound_positive'] and diag['rho_lambda_min'] > -1e-12
    out = {
        **base,
        **whole,
        'node_bound_file': str(a.certified_node_lower_bounds),
        'node_bound_file_sha256': sha256(a.certified_node_lower_bounds),
        'protecting_gap_certified': bool(whole['uniform_bound_positive']),
        'status': 'PASS_LOCALIZED_RIESZ_KATO_PROJECTOR_WITH_UNIFORM_CONTOUR_BOUND' if ok else 'RIZSZ_KATO_PROJECTOR_NOT_CERTIFIED',
    }
    a.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')

    if not ok:
        print(json.dumps(out, indent=2))
        return

    a.out_density.parent.mkdir(parents=True, exist_ok=True)
    gram=P/np.sqrt(float(np.trace(P@P.conj().T).real))
    np.savez_compressed(
        a.out_density,
        rho=rho,
        rho_gram_factor=gram,
        buffer=np.int64(-1),
        uniform_contour_gap_lower_bound=np.float64(whole['uniform_contour_singular_value_lower_bound']),
        riesz_residual_upper_bound=np.float64(max(diag['projector_idempotency_fro_diagnostic'], diag['projector_commutator_fro_diagnostic'])),
        contour_sample_minimum_diagnostic=np.float64(np.min(sampled)),
        contour_interpolation_loss=np.float64(whole['contour_lipschitz_interpolation_loss']),
    )
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
