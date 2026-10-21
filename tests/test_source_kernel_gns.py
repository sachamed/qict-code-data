"""Independent finite checks of the algebraic-source formulation."""
import sys
import unittest
from fractions import Fraction
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
from source_kernel_gns import (normalize_source_kernel, block_toeplitz,
                               normalized_kernel_error, bilateral_shift_control)


class AlgebraicSourceTests(unittest.TestCase):
    def test_shift_has_six_stationary_source_atoms(self):
        kernel = bilateral_shift_control(21, 5)
        moments, diagnostic = normalize_source_kernel(kernel)
        phases = -2*np.pi*np.arange(1, 7)/7
        for n in range(-5, 6):
            np.testing.assert_allclose(moments[n], np.diag(np.exp(1j*n*phases)),
                                       rtol=0, atol=2e-15)
        self.assertFalse(diagnostic["physical_state_selected"])
        self.assertFalse(diagnostic["infinite_volume_limit_certified"])
        self.assertGreater(np.linalg.eigvalsh(block_toeplitz(moments, 5))[0], -1e-13)

    def test_trace_norm_fails_while_source_kernels_stabilize(self):
        for t in (7, 14, 28, 56):
            rho = np.zeros(2*t); rho[:t] = 1/t
            rho2 = np.full(2*t, 1/(2*t))
            exact_distance = t*abs(Fraction(1,t)-Fraction(1,2*t))+t*Fraction(1,2*t)
            self.assertEqual(exact_distance, 1)
            self.assertAlmostEqual(float(np.sum(abs(rho-rho2))), 1.0, places=14)
            first = bilateral_shift_control(t, 3)
            second = bilateral_shift_control(2*t, 3)
            for n in first:
                np.testing.assert_array_equal(first[n], second[n])

    def test_incomplete_average_is_not_assumed_stationary(self):
        raw = bilateral_shift_control(8, 2)
        with self.assertRaisesRegex(ValueError, "reflection"):
            normalize_source_kernel(raw)

    def test_stationary_density_matches_relative_hilbert_schmidt(self):
        rng = np.random.default_rng(2163)
        u0 = np.diag(np.exp(1j*np.array([.2, .5, -.7])))
        q, _ = np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
        uf = q @ np.diag(np.exp(1j*np.array([.1, -.4, .8, 1.1]))) @ q.conj().T
        rho = np.diag([.2, .3, .5]); root = np.diag(np.sqrt([.2, .3, .5]))
        sources = rng.normal(size=(6, 4, 3))+1j*rng.normal(size=(6, 4, 3))
        raw = {}
        source_columns = np.column_stack([(o @ root).reshape(-1, order="F")
                                          for o in sources])
        relative = np.kron(u0.conj(), uf)
        for n in range(-3, 4):
            raw[n] = np.array([[np.trace(rho @ a.conj().T
                           @ np.linalg.matrix_power(uf, n) @ b
                           @ np.linalg.matrix_power(u0, -n))
                           for b in sources] for a in sources])
            expected = source_columns.conj().T @ np.linalg.matrix_power(relative, n) @ source_columns
            np.testing.assert_allclose(raw[n], expected, rtol=2e-14, atol=2e-14)
        moments, _ = normalize_source_kernel(raw)
        np.testing.assert_allclose(moments[0], np.eye(6), rtol=0, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(block_toeplitz(moments, 3))[0], -1e-13)

    def test_normalized_error_bound_with_changed_gram(self):
        rng = np.random.default_rng(630)
        a = rng.normal(size=(9, 6))+1j*rng.normal(size=(9, 6))
        b = a+.001*(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape))
        unitary = np.diag(np.exp(1j*np.linspace(-1, 1, 9)))
        ka = {n: a.conj().T @ np.linalg.matrix_power(unitary, n) @ a for n in (-1, 0, 1)}
        kb = {n: b.conj().T @ np.linalg.matrix_power(unitary, n) @ b for n in (-1, 0, 1)}
        ca, _ = normalize_source_kernel(ka); cb, _ = normalize_source_kernel(kb)
        gamma = min(np.linalg.eigvalsh(ka[0])[0], np.linalg.eigvalsh(kb[0])[0])
        bound = normalized_kernel_error(np.linalg.norm(ka[1]-kb[1], 2),
                    np.linalg.norm(ka[0]-kb[0], 2), gamma, np.linalg.norm(ka[1], 2))
        self.assertLessEqual(np.linalg.norm(ca[1]-cb[1], 2), bound)

    def test_singular_gram_is_not_repaired(self):
        with self.assertRaisesRegex(ValueError, "singular"):
            normalize_source_kernel({0: np.diag([1., 0.])})

    def test_missing_lags_and_nonfinite_data_rejected(self):
        for kernel in ({1: np.eye(2)}, {0: np.eye(2), 1: np.eye(2)},
                       {0: np.array([[float("nan")]])}):
            with self.assertRaises(ValueError):
                normalize_source_kernel(kernel)
        with self.assertRaises(ValueError):
            block_toeplitz({0: np.eye(2)}, 1)

    def test_invalid_error_bounds_rejected(self):
        for args in ((1, 1, 0, 1), (-1, 1, 1, 1), (1, 1, 1, float("inf"))):
            with self.assertRaises(ValueError):
                normalized_kernel_error(*args)


if __name__ == "__main__":
    unittest.main()
