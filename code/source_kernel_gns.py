"""Normalize stationary source kernels without requiring a density matrix.

The caller supplies K[n]_{ab}=omega(O_a^* alpha^n(O_b)) on one source
algebra. These finite linear-algebra checks do not establish the existence,
selection, convergence, or physical vacuum interpretation of that omega.
Floating diagnostics are not interval certificates.
"""
from __future__ import annotations

import math
import numpy as np


def normalize_source_kernel(kernel, tolerance=1e-11):
    """Return G^(-1/2) K[n] G^(-1/2) and explicit numerical diagnostics.

    All lags must include their negatives. No source is deleted and no small
    Gram eigenvalue is clipped. The Hermitian part of K[0] is used solely
    after its antisymmetric residual has passed the requested tolerance;
    this adjustment is reported. Positivity of all orders is not inferred
    from a finite list of moments.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if 0 not in kernel or not kernel:
        raise ValueError("zero-lag Gram matrix is required")
    values = {}
    for n, value in kernel.items():
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise ValueError("lags must be integers")
        a = np.asarray(value, dtype=complex)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.shape[0]:
            raise ValueError("kernel entries must be nonempty square matrices")
        if not np.isfinite(a).all():
            raise ValueError("kernel entries must be finite")
        values[int(n)] = a.copy()
    shape = values[0].shape
    if any(a.shape != shape for a in values.values()):
        raise ValueError("source dimension must be common to all lags")
    if any(-n not in values for n in values):
        raise ValueError("every lag must have its negative")
    reflection = max(float(np.linalg.norm(values[-n]-a.conj().T, 2))
                     for n, a in values.items())
    if reflection > tolerance:
        raise ValueError("kernel violates Hermitian reflection")
    gram = (values[0]+values[0].conj().T)/2
    eigenvalues, vectors = np.linalg.eigh(gram)
    if eigenvalues[0] <= tolerance:
        raise ValueError("Gram matrix is singular or numerically unresolved")
    inverse_sqrt = (vectors/np.sqrt(eigenvalues)) @ vectors.conj().T
    normalized = {n: inverse_sqrt @ a @ inverse_sqrt for n, a in values.items()}
    return normalized, {
        "classification": "FINITE_SOURCE_KERNEL_DIAGNOSTIC",
        "source_count": shape[0],
        "gram_min_eigenvalue_diagnostic": float(eigenvalues[0]),
        "gram_hermitization_norm": float(np.linalg.norm(gram-values[0], 2)),
        "reflection_residual": reflection,
        "C0_residual": float(np.linalg.norm(normalized[0]-np.eye(shape[0]), 2)),
        "physical_state_selected": False,
        "infinite_volume_limit_certified": False,
        "roundoff_certified": False,
    }


def block_toeplitz(moments, depth):
    """Build [C_(j-i)]_(i,j=0)^depth; missing lags are rejected."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    if any(n not in moments for n in range(-depth, depth+1)):
        raise ValueError("insufficient moment depth")
    return np.block([[moments[j-i] for j in range(depth+1)]
                     for i in range(depth+1)])


def normalized_kernel_error(epsilon_n, epsilon_0, gamma, kernel_norm_bound):
    """Evaluate epsilon_n/gamma+B*epsilon_0/gamma**2, excluding rounding.

    Assumptions: both exact Gram matrices >= gamma I, raw kernel difference
    <= epsilon_n, Gram difference <= epsilon_0, and ||K[n]|| <= B.
    These bounds must be supplied independently, not estimated here.
    """
    args = (epsilon_n, epsilon_0, gamma, kernel_norm_bound)
    if any(not math.isfinite(x) for x in args):
        raise ValueError("all bounds must be finite")
    if gamma <= 0 or min(epsilon_n, epsilon_0, kernel_norm_bound) < 0:
        raise ValueError("positive Gram gap and nonnegative errors are required")
    return epsilon_n/gamma + kernel_norm_bound*epsilon_0/gamma**2


def bilateral_shift_control(average_steps, depth, period=7):
    """Auxiliary infinite-shift kernels, not physical QICT data.

    S|j>=|j+1>, D_a|j>=exp(2*pi*i*a*j/period)|j>, a=1,...,period-1.
    rho_T=T^-1 sum_(j=0)^(T-1)|j><j|. Return its raw source kernels
    K_ab(n)=exp(-2*pi*i*b*n/period) omega_T(D_a^* D_b).
    With T a multiple of period, these kernels are exactly stationary in
    the algebraic calculation, despite ||rho_T-rho_(2T)||_1=1.
    """
    for value in (average_steps, depth, period):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("integer parameters are required")
    if average_steps <= 0 or depth < 0 or period < 2:
        raise ValueError("invalid averaging length, depth, or period")
    indices = np.arange(1, period)
    phases = 2*np.pi*indices/period
    # Complete periods cancel exactly; sum only the incomplete remainder.
    full, remainder = divmod(average_steps, period)
    gram = (full*period/average_steps)*np.eye(period-1, dtype=complex)
    for j in range(remainder):
        v = np.exp(1j*phases*j)
        gram += np.outer(v.conj(), v)/average_steps
    return {n: gram @ np.diag(np.exp(-1j*phases*n))
            for n in range(-depth, depth+1)}
