"""Deterministic finite-dimensional checks of exact receiver-moment identities.

Small test matrices, independent of the physical QICT G2 contraction.
"""
from __future__ import annotations
import numpy as np


def unitary_from_hermitian(h: np.ndarray, t: float) -> np.ndarray:
    w, z = np.linalg.eigh(h)
    return (z * np.exp(1j * t * w)) @ z.conj().T


def main() -> None:
    rng = np.random.default_rng(1947)
    q = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    h = (q + q.conj().T) / 2
    h = h / np.linalg.norm(h, 2)
    a = np.diag(np.linspace(-0.28, 0.31, 8))
    q0 = rng.standard_normal((8, 6)) + 1j * rng.standard_normal((8, 6))
    v0, _ = np.linalg.qr(q0)
    u0 = unitary_from_hermitian(a, 1.0)
    i6 = np.eye(6)

    def cv(t: float, n: int) -> np.ndarray:
        u = unitary_from_hermitian(a, 1.0+t)
        v = unitary_from_hermitian(h, t) @ v0
        return v.conj().T @ np.linalg.matrix_power(u, n) @ v

    c1 = cv(0, 1)
    dmat = 2 * i6 - c1 - c1.conj().T
    x = (u0 - np.eye(8)) @ v0
    assert np.linalg.norm(dmat - x.conj().T @ x) < 1e-12
    assert np.min(np.linalg.eigvalsh(dmat)) > -1e-12
    d = np.sqrt(np.linalg.norm(dmat, 2))
    for n in range(1, 9):
        higher_defect = 2 * i6 - cv(0,n) - cv(0,n).conj().T
        assert np.min(np.linalg.eigvalsh(higher_defect)) > -1e-11
        assert np.min(np.linalg.eigvalsh(n*n*dmat-higher_defect)) > -1e-10
        for m in range(n):
            assert np.linalg.norm(cv(0, n) - cv(0, m), 2) < (n-m)*d+1e-12

    up = 1j * a @ u0
    upp = -(a @ a) @ u0
    vp = 1j * h @ v0
    vpp = -(h @ h) @ v0
    n = 5
    uns = [np.linalg.matrix_power(u0, k) for k in range(n+1)]
    u_n_prime = sum((uns[j] @ up @ uns[n-1-j] for j in range(n)),np.zeros((8,8),complex))
    u_n_second = sum((uns[j] @ upp @ uns[n-1-j] for j in range(n)),np.zeros((8,8),complex))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Select the second derivative factors at two distinct positions.
            factors = [u0] * n
            factors[i], factors[j] = up, up
            tmp = np.eye(8,dtype=complex)
            for factor in factors:
                tmp = tmp @ factor
            u_n_second += tmp

    cprime = vp.conj().T @ uns[n] @ v0 + v0.conj().T @ uns[n] @ vp + v0.conj().T @ u_n_prime @ v0
    csecond = (vpp.conj().T @ uns[n] @ v0 + v0.conj().T @ uns[n] @ vpp
               + 2*vp.conj().T @ uns[n] @ vp + 2*vp.conj().T @ u_n_prime @ v0
               + 2*v0.conj().T @ u_n_prime @ vp + v0.conj().T @ u_n_second @ v0)
    hstep = 1e-4
    numeric_first = (cv(hstep, n) - cv(-hstep, n))/(2*hstep)
    numeric_second = (cv(hstep, n) - 2*cv(0,n) + cv(-hstep,n))/hstep**2
    assert np.linalg.norm(cprime-numeric_first,2) < 2e-6
    assert np.linalg.norm(csecond-numeric_second,2) < 2e-5
    bound1=2*np.linalg.norm(vp,2)+n*np.linalg.norm(up,2)
    bound2=(2*np.linalg.norm(vpp,2)+2*np.linalg.norm(vp,2)**2
            +4*n*np.linalg.norm(vp,2)*np.linalg.norm(up,2)
            +n*np.linalg.norm(upp,2)+n*(n-1)*np.linalg.norm(up,2)**2)
    assert np.linalg.norm(cprime,2)<=bound1+1e-12
    assert np.linalg.norm(csecond,2)<=bound2+1e-12
    assert np.linalg.norm(v0.conj().T@v0 - i6)<1e-12
    print('PASS: Gram defect, full-depth speed bounds, two moment derivatives and their estimates')

if __name__ == '__main__':
    main()
