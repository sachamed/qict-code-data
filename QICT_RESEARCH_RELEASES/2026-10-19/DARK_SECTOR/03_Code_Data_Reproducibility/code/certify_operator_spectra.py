"""Finite C9 spectral separation with outward-rounded binary64 arithmetic.

The original arrays remain unchanged. The exactly defined unitary model uses
the polar factor of each stored square matrix. Schur decompositions propose
eigenvalue centres; interval residuals certify disks containing the spectrum.
All tensor-product centres are checked, including transitions of zero weight.
The certificate uses only elementary IEEE-754 operations and nextafter; BLAS
and LAPACK are used for proposals, never for the interval verification.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import schur

ROOT = Path(__file__).resolve().parents[1]


def down(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


def add(a, b):
    return down(a[0] + b[0]), up(a[1] + b[1])


def neg(a):
    return -a[1], -a[0]


def mul(a, b):
    p = [a[i] * b[j] for i in (0, 1) for j in (0, 1)]
    return down(np.minimum.reduce(p)), up(np.maximum.reduce(p))


def exact(z):
    z = np.asarray(z)
    return ((z.real, z.real), (z.imag, z.imag))


def cmul(a, b):
    return add(mul(a[0], b[0]), neg(mul(a[1], b[1]))), add(
        mul(a[0], b[1]), mul(a[1], b[0]))


def csub(a, b):
    return add(a[0], neg(b[0])), add(a[1], neg(b[1]))


def cmatmul(A, B):
    """Enclose the exact product of the binary complex matrices A and B."""
    shape = (A.shape[0], B.shape[1])
    out = ((np.zeros(shape), np.zeros(shape)),
           (np.zeros(shape), np.zeros(shape)))
    for k in range(A.shape[1]):
        term = cmul(exact(A[:, k, None]), exact(B[None, k, :]))
        out = add(out[0], term[0]), add(out[1], term[1])
    return out


def sum_up(x, axis):
    x = np.moveaxis(np.asarray(x), axis, 0)
    out = np.zeros(x.shape[1:])
    for row in x:
        out = up(out + row)
    return out


def norm_bound(a):
    """Upper bound max(||A||_1,||A||_infinity), hence ||A||_2."""
    ar = np.maximum(abs(a[0][0]), abs(a[0][1]))
    ai = np.maximum(abs(a[1][0]), abs(a[1][1]))
    absolute = up(ar + ai)
    return float(max(np.max(sum_up(absolute, 0)),
                     np.max(sum_up(absolute, 1))))


def spectral_disks(A):
    T, Q = schur(A, output='complex')
    centres = np.diag(T).copy()
    identity = exact(np.eye(len(A), dtype=complex))
    da = norm_bound(csub(cmatmul(A.conj().T, A), identity))
    dq = norm_bound(csub(cmatmul(Q.conj().T, Q), identity))
    residual = norm_bound(csub(cmatmul(A, Q),
                              cmul(exact(Q), exact(centres[None, :]))))
    if not (da < .5 and dq < .5 and np.max(abs(centres)) < 1.9):
        raise ArithmeticError('spectral-disk hypotheses failed')
    # ||polar(A)-A|| <= da; ||Q||,||Q^-1|| <= 1+dq.
    qbound = up(1. + dq)
    radius = float(up(qbound * up(residual + up(da * qbound))))
    # Verify |centre|<2 using |Re|+|Im|, with outward addition.
    assert np.max(up(abs(centres.real)+abs(centres.imag))) < 2.
    return centres, Q, {
        'dimension': len(A), 'unitarity_defect_upper': da,
        'basis_unitarity_defect_upper': dq,
        'residual_norm_upper': residual,
        'polar_spectral_disk_radius_upper': radius,
    }


def abs_lower(a):
    return np.where(a[0] > 0., a[0],
                    np.where(a[1] < 0., -a[1], 0.))


def distance_lower(z):
    """Lower bound |z-1| via max(|Re(z)-1|,|Im(z)|)."""
    real = add(z[0], (-1., -1.))
    return np.maximum(abs_lower(real), abs_lower(z[1]))


def pair_centres(a, b):
    return cmul(exact(a[:, None]), exact(b.conj()[None, :]))


def flatten(z):
    return tuple(tuple(v.ravel() for v in part) for part in z)


def separated_tensor_pairs(first, second, radii):
    """Check every 4-factor centre in bounded-memory chunks."""
    first, second = flatten(first), flatten(second)
    minimum = np.inf
    count = first[0][0].size * second[0][0].size
    for start in range(0, first[0][0].size, 64):
        a = tuple(tuple(v[start:start+64, None] for v in p) for p in first)
        b = tuple(tuple(v[None, :] for v in p) for p in second)
        minimum = min(minimum, float(np.min(distance_lower(cmul(a, b)))))
    # Product telescoping: each exact eigenvalue has modulus one, each
    # centre modulus <2, so the four-factor error is <=8*sum(radii).
    error = float(up(8. * sum_up(np.asarray(radii), 0)))
    gap = float(down(minimum - error))
    assert gap > 0.
    return {'candidate_count': count, 'centre_distance_lower': minimum,
            'product_error_upper': error, 'unitary_chord_gap_lower': gap,
            'zero_phase_atom_weight': 0}


def main():
    path = ROOT/'code/FINITE_C9_COMMON_STATE_AND_MOMENTS.npz'
    data = np.load(path, allow_pickle=False)
    output = {'model': 'Tensor products of the exact polar factors of the stored matrices',
              'arithmetic': 'Outward-rounded IEEE-754 binary64 elementary operations',
              'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'regulators': []}
    for r in range(1, 5):
        z, diagnostics = {}, {}
        for name in ['neutral_U1', 'left_U1', 'right_U1',
                     'neutral_SU2', 'charged_SU2']:
            z[name], _, diagnostics[name] = spectral_disks(data[f'R{r}_{name}'])
        radii = {key: d['polar_spectral_disk_radius_upper']
                 for key, d in diagnostics.items()}
        source = data[f'R{r}_weak_source'].astype(complex)
        source_defect = norm_bound(csub(cmatmul(source.conj().T, source),
                                       exact(np.eye(source.shape[1], dtype=complex))))
        assert source_defect < .5
        right_centres = pair_centres(z['right_U1'], z['neutral_U1'])
        right_lower = float(np.min(distance_lower(right_centres)))
        right_error = float(up(2. * up(radii['right_U1']+radii['neutral_U1'])))
        right_gap = float(down(right_lower-right_error))
        assert right_gap > 0.
        left = separated_tensor_pairs(
            pair_centres(z['left_U1'], z['neutral_U1']),
            pair_centres(z['charged_SU2'], z['neutral_SU2']),
            [radii[n] for n in ['left_U1', 'neutral_U1', 'charged_SU2', 'neutral_SU2']])
        row = {'regulator': f'R{r}', 'matrix_certificates': diagnostics,
               'weak_source_isometry_defect_upper': source_defect,
               'left': left, 'right': {
                   'candidate_count': len(z['right_U1'])*len(z['neutral_U1']),
                   'centre_distance_lower': right_lower,
                   'product_error_upper': right_error,
                   'unitary_chord_gap_lower': right_gap,
                   'zero_phase_atom_weight': 0}}
        output['regulators'].append(row)
        print(json.dumps({'regulator': row['regulator'], 'left': left,
                          'right': row['right']}), flush=True)
    target = ROOT/'data/OPERATOR_SPECTRAL_CERTIFICATE.json'
    target.write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
