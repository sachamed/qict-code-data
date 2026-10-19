"""Defensive numerical *screening* for QICT data, never physical certification.

Successful tests do not establish the global interacting operator, a Kato limit,
or a physical pole. They only reject malformed finite-dimensional inputs.
"""
import numpy as np

SOURCE_ORDER = ('L1', 'L2', 'L3', 'R1', 'R2', 'R3')


def _array(value, name):
    try:
        array = np.asarray(value, dtype=np.complex128)
        if not np.isfinite(array).all():
            raise ValueError(f'{name} has NaN or infinite entries')
        return array
    except (TypeError, OverflowError) as exc:
        raise ValueError(f'Invalid {name} entries') from exc


def _tolerance(value):
    if not np.isfinite(value) or not (0 < value < 1):
        raise ValueError('Tolerance must be finite and between zero and one')


def check_density(rho, factor, tol=1e-9):
    _tolerance(tol)
    rho = _array(rho, 'rho')
    factor = _array(factor, 'rho_gram_factor')
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or not rho.shape[0]:
        raise ValueError('Invalid Kato density dimensions')
    if factor.ndim != 2 or factor.shape[0] != rho.shape[0]:
        raise ValueError('Density and Gram-factor dimensions disagree')
    if rho.shape[0] > 4096:
        raise ValueError('Requires a separate large-scale spectral certificate')
    gram = factor @ factor.conj().T
    if not np.isfinite(gram).all():
        raise ValueError('Gram factor overflow')
    tr = float(gram.trace().real)
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError('Degenerate or nonfinite Gram normalization')
    gram /= tr
    try:
        if np.linalg.norm(rho - gram) > tol:
            raise ValueError('Gram-factor identity fails')
        if np.linalg.norm(rho - rho.conj().T) > tol:
            raise ValueError('Density is not Hermitian')
        if abs(rho.trace() - 1) > tol:
            raise ValueError('Density is not trace one')
        if np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() < -tol:
            raise ValueError('Density is not positive')
    except np.linalg.LinAlgError as exc:
        raise ValueError('Ill-conditioned density screening') from exc
    return {'status': 'DIAGNOSTIC_ONLY', 'dimension': int(rho.shape[0])}


def source_isometry(V, tol=1e-10):
    _tolerance(tol)
    V = _array(V, 'rank-six source')
    if V.ndim != 2 or V.shape[1] != 6 or V.shape[0] < 6:
        raise ValueError('Incorrect rank-six source dimensions')
    try:
        deviation = float(np.linalg.norm(V.conj().T @ V - np.eye(6), ord=2))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Ill-conditioned rank-six source') from exc
    if not np.isfinite(deviation) or deviation > tol:
        raise ValueError('Rank-six source isometry fails')
    return deviation


def block_toeplitz(C):
    C = _array(C, 'CN6 moments')
    if C.ndim != 3 or C.shape[1:] != (6, 6) or len(C) < 2:
        raise ValueError('Requires at least two six-by-six moments')
    n = len(C)
    if n * 6 > 2500:
        raise ValueError('Use an independently certified sparse Toeplitz bound')
    T = np.zeros((6 * n, 6 * n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            k = i - j
            T[6 * i:6 * i + 6, 6 * j:6 * j + 6] = C[k] if k >= 0 else C[-k].conj().T
    return T


def moment_diagnostic(C, errors, source_order=SOURCE_ORDER, tol=1e-10):
    _tolerance(tol)
    C = _array(C, 'CN6 moments')
    try:
        errors = np.asarray(errors, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid moment-error vector') from exc
    if tuple(source_order) != SOURCE_ORDER:
        raise ValueError('Incorrect six-source order')
    if C.ndim != 3 or C.shape[1:] != (6, 6) or len(C) < 2:
        raise ValueError('CN6 moments have incorrect dimensions')
    if errors.shape != (len(C),):
        raise ValueError('Moment-error shape mismatch')
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('Moment error bounds are invalid')
    if np.linalg.norm(C[0] - np.eye(6), ord=2) > tol:
        raise ValueError('C0 != I6')
    T = block_toeplitz(C)
    try:
        eigenmin = float(np.linalg.eigvalsh((T + T.conj().T) / 2)[0])
    except np.linalg.LinAlgError as exc:
        raise ValueError('Failed finite Toeplitz eigenscreen') from exc
    bound = float(errors[0] + 2 * errors[1:].sum())
    if not np.isfinite(eigenmin) or not np.isfinite(bound):
        raise ValueError('Nonfinite Toeplitz diagnostic')
    return {'status': 'DIAGNOSTIC_ONLY', 'toeplitz_float64_min': eigenmin,
            'reported_error_norm_upper': bound, 'certified_psd': False}