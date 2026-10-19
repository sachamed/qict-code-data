import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from validate_physical_inputs import (block_toeplitz, check_density, moment_diagnostic,
                                      source_isometry)


class NumericalRefusalTests(unittest.TestCase):
    def test_finite_isometry(self):
        self.assertLess(source_isometry(np.eye(6)), 1e-15)

    def test_nan_source_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            source = np.eye(6)
            source[0, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                source_isometry(source)

    def test_complex_nan_source_rejected(self):
        value = np.eye(6, dtype=complex)
        value[0, 1] = np.nan * 1j
        with self.assertRaises(ValueError):
            source_isometry(value)

    def test_malformed_source_dimension(self):
        with self.assertRaises(ValueError):
            source_isometry(np.zeros((5, 6)))

    def test_bad_density_rejected(self):
        for bad in (np.nan, np.inf):
            rho = np.eye(2) / 2
            rho[0, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_density(rho, np.eye(2))

    def test_valid_density_screened_not_certified(self):
        self.assertEqual(check_density(np.eye(2)/2, np.eye(2))['status'], 'DIAGNOSTIC_ONLY')

    def test_invalid_moments_rejected(self):
        C = np.stack([np.eye(6), np.eye(6)])
        C[1, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            moment_diagnostic(C, [0, 0])

    def test_moment_error_invalid_rejected(self):
        C = np.stack([np.eye(6), np.eye(6)])
        with self.assertRaises(ValueError):
            moment_diagnostic(C, [0, np.nan])
        with self.assertRaises(ValueError):
            moment_diagnostic(C, [0, -0.1])

    def test_valid_moments_are_screen_only(self):
        C = np.stack([np.eye(6), np.eye(6)])
        result = moment_diagnostic(C, [0, 0])
        self.assertEqual(result['status'], 'DIAGNOSTIC_ONLY')
        self.assertFalse(result['certified_psd'])

    def test_moment_source_order(self):
        C = np.stack([np.eye(6), np.eye(6)])
        with self.assertRaises(ValueError):
            moment_diagnostic(C, [0, 0], ['L2', 'L1', 'L3', 'R1', 'R2', 'R3'])

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            source_isometry(np.eye(6), float('nan'))
        with self.assertRaises(ValueError):
            check_density(np.eye(2)/2, np.eye(2), tol=-1)


if __name__ == '__main__':
    unittest.main()