"""Independent rational checks of the interval arithmetic and spectral bound."""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
from certify_operator_spectra import (cmatmul, distance_lower, exact, mul,
                                     pair_centres, spectral_disks)

ROOT = Path(__file__).resolve().parents[1]


def rational_complex(z):
    return F(float(z.real)), F(float(z.imag))


def product(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def main():
    assert np.finfo(np.float64).bits == 64
    assert np.nextafter(0., 1.) > 0.
    rng = np.random.default_rng(49103)
    checked = 0
    for shape in [(3, 4, 2), (7, 11, 5), (2, 37, 3)]:
        n, k, m = shape
        A = rng.normal(size=(n, k))+1j*rng.normal(size=(n, k))
        B = rng.normal(size=(k, m))+1j*rng.normal(size=(k, m))
        intervals = cmatmul(A, B)
        for i in range(n):
            for j in range(m):
                terms = [product(rational_complex(A[i, q]), rational_complex(B[q, j]))
                         for q in range(k)]
                exact_value = tuple(sum(t[s] for t in terms) for s in (0, 1))
                for s in (0, 1):
                    assert F(float(intervals[s][0][i, j])) <= exact_value[s]
                    assert exact_value[s] <= F(float(intervals[s][1][i, j]))
                checked += 1
    # Subnormal and cancellation cases, checked against exact rational values.
    for x, y in [(1e-200, 1e-200), (1., -1.), (2.**-1022, .5)]:
        lo, hi = mul((x, x), (y, y))
        assert F(float(lo)) <= F(x)*F(y) <= F(float(hi))
    # Known exactly unitary diagonal models exercise zero and nonzero phases.
    A = np.diag([1.+0j, 1j, -1.+0j])
    z, _, d = spectral_disks(A)
    assert max(min(abs(z-x)) for x in np.diag(A)) <= d['polar_spectral_disk_radius_upper']
    zero_centres = pair_centres(z, z)
    assert np.min(distance_lower(zero_centres)) == 0.
    # A scaled planar rotation has an explicitly known polar spectrum.
    B = np.array([[.6, -.8], [.8, .6]], dtype=complex)
    zz, _, bd = spectral_disks(B)
    # Binary stored .6 and .8 share one common radial polar normalization.
    eigenvalues = np.array([.6+.8j, .6-.8j])/np.hypot(.6, .8)
    assert max(min(abs(zz-x)) for x in eigenvalues) <= bd['polar_spectral_disk_radius_upper']
    cert = json.loads((ROOT/'data/OPERATOR_SPECTRAL_CERTIFICATE.json').read_text())
    for row in cert['regulators']:
        for group in ('left', 'right'):
            assert row[group]['unitary_chord_gap_lower'] > 0.
            assert row[group]['zero_phase_atom_weight'] == 0
    output = {'exact_rational_matrix_entries_checked': checked,
              'subnormal_interval_checks': 'passed',
              'known_spectrum_and_zero_phase_controls': 'passed',
              'four_regulator_separation_checks': 'passed'}
    (ROOT/'data/OPERATOR_CERTIFICATE_VALIDATION.json').write_text(
        json.dumps(output, indent=2, sort_keys=True)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
