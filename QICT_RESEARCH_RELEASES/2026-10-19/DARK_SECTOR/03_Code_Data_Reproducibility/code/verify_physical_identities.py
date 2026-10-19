"""Independent numerical controls for the supplementary mathematical identities.

These dimensionless examples test formulas; they are not QICT dark-particle
parameters. Exact scalar cumulants are compared with an independently inverted
source map, and spectral projectors with numerical contour integration.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]


def pole_checks():
    centre, radius = 0.75, 0.3
    h = np.diag([centre, 2.0, 4.0])
    v = np.array([1.0, 0.0, 0.0])
    projector = np.outer(v, v)
    identity = np.eye(3)
    result = []
    for regulator in (2, 4, 8, 16, 32):
        angle = 0.1 / regulator
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        hr = rotation @ np.diag([centre + 0.01 / regulator, 2, 4]) @ rotation.T
        energies, states = np.linalg.eigh(hr)
        pr = np.outer(states[:, 0], states[:, 0])
        perturbation = np.linalg.norm(hr - h, 2)
        assert perturbation < radius
        # Resolvent identity and the distance of the contour from each spectrum.
        eta = perturbation / (radius * (radius - perturbation))
        projection_bound = radius * eta
        energy_bound = 2 * radius**2 * eta
        projection_error = np.linalg.norm(pr - projector, 2)
        energy_error = abs(energies[0] - centre)
        residue_error = abs(v @ pr @ v - 1)
        contour = np.zeros((3, 3), dtype=complex)
        for theta in 2 * np.pi * (np.arange(256) + 0.5) / 256:
            dz = radius * np.exp(1j * theta)
            contour += dz * np.linalg.inv((centre + dz) * identity - hr) / 256
        contour_error = np.linalg.norm(contour - pr, 2)
        assert projection_error <= projection_bound
        assert energy_error <= energy_bound
        assert residue_error <= projection_bound
        assert contour_error < 2e-14
        # Exercise both terms in the source-residue enclosure with a convergent
        # source family distinct from the reference source used above.
        source_angle = 0.03 / regulator
        vr = np.array([np.cos(source_angle), np.sin(source_angle), 0.0])
        source_error = np.linalg.norm(vr - v)
        computed_residue = float(vr @ pr @ vr)
        full_residue_bound = projection_bound + 2 * source_error
        visibility_lower_bound = computed_residue - full_residue_bound
        amplitude_delta = projection_bound + source_error
        amplitude_lower = max(0.0, np.sqrt(computed_residue) - amplitude_delta)**2
        amplitude_upper = min(1.0, (np.sqrt(computed_residue) + amplitude_delta)**2)
        assert abs(computed_residue - 1.0) <= full_residue_bound
        assert 0 < visibility_lower_bound <= 1.0
        assert 0 < amplitude_lower <= 1.0 <= amplitude_upper + 2e-15
        # The same isolated eigenvalue is invisible to an orthogonal source.
        invisible_source = np.array([0.0, 0.0, 1.0])
        invisible_residue = float(invisible_source @ pr @ invisible_source)
        assert invisible_residue == 0.0
        for z in (centre + 0.05j, centre + 0.1j):
            source_resolvent = invisible_source @ np.linalg.solve(
                z * identity - hr, invisible_source)
            assert abs(source_resolvent - 1 / (z - 4.0)) < 2e-15
        # Shift both operators and the contour. Physical errors and the centred
        # bound are invariant; independent eigensolves and contour quadrature
        # also exercise negative and large positive energy origins.
        origin_checks = []
        for offset in (-7.5, 80.0):
            shifted_h = hr + offset * identity
            shifted_energy = np.linalg.eigvalsh(shifted_h)[0]
            shifted_error = abs(shifted_energy - (centre + offset))
            shifted_contour = np.zeros((3, 3), dtype=complex)
            for theta in 2 * np.pi * (np.arange(256) + 0.5) / 256:
                dz = radius * np.exp(1j * theta)
                shifted_contour += dz * np.linalg.inv(
                    (centre + offset + dz) * identity - shifted_h) / 256
            drift = abs(shifted_error - energy_error)
            shifted_contour_error = np.linalg.norm(shifted_contour - pr, 2)
            assert drift < 2e-13 and shifted_error <= energy_bound
            assert shifted_contour_error < 2e-13
            origin_checks.append(dict(energy_offset=offset,
                                      energy_error_drift=drift,
                                      contour_error=shifted_contour_error))
        result.append(dict(regulator=regulator, projection_error=projection_error,
                           projection_bound=projection_bound, energy_error=energy_error,
                           energy_bound=energy_bound, residue_error=residue_error,
                           residue_bound=projection_bound, contour_error=contour_error,
                           energy_origin_checks=origin_checks,
                           convergent_source_error=source_error,
                           convergent_source_residue=computed_residue,
                           full_source_residue_bound=full_residue_bound,
                           visibility_lower_bound=visibility_lower_bound,
                           amplitude_delta=amplitude_delta,
                           amplitude_lower_bound=amplitude_lower,
                           amplitude_upper_bound=amplitude_upper,
                           orthogonal_source_residue=invisible_residue))
    return result


def weak_source_check():
    """Independent eigensolve with small overlap, not a QICT particle model."""
    h = np.diag([0.75, 2.0, 4.0])
    angle, radius = 0.001, 0.3
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    hr = rotation @ h @ rotation.T
    _, states = np.linalg.eigh(hr)
    source = np.array([0.02, np.sqrt(1 - 0.02**2), 0.0])
    pr = np.outer(states[:, 0], states[:, 0])
    p = np.diag([1.0, 0.0, 0.0])
    limiting_residue = float(source @ p @ source)
    finite_residue = float(source @ pr @ source)
    perturbation = np.linalg.norm(hr - h, 2)
    assert perturbation < radius
    eta = perturbation / (radius * (radius - perturbation))
    delta = radius * eta
    linear_lower = finite_residue - delta
    amplitude_lower = max(0.0, np.sqrt(finite_residue) - delta)**2
    amplitude_upper = min(1.0, (np.sqrt(finite_residue) + delta)**2)
    assert linear_lower < 0 < amplitude_lower <= limiting_residue <= amplitude_upper
    return dict(finite_residue=finite_residue, limiting_residue=limiting_residue,
                resolvent_error_bound=eta, amplitude_delta=delta,
                linear_lower_bound=linear_lower,
                amplitude_lower_bound=amplitude_lower,
                amplitude_upper_bound=amplitude_upper, status='PASS')


def legendre_checks():
    outcomes, probabilities = [F(0), F(1), F(3)], [F(1, 7), F(2, 7), F(4, 7)]
    mean = sum(p * x for p, x in zip(probabilities, outcomes))
    mu = {n: sum(p * (x - mean)**n for p, x in zip(probabilities, outcomes))
          for n in (2, 3, 4)}
    k2, k3, k4 = mu[2], mu[3], mu[4] - 3 * mu[2]**2
    gamma3 = -k3 / k2**3
    gamma4 = 3 * k3**2 / k2**5 - k4 / k2**4
    x = np.array(outcomes, dtype=float)
    p = np.array(probabilities, dtype=float)

    def mean_at_source(source):
        weight = p * np.exp(source * x)
        return float(np.dot(weight, x) / weight.sum())

    shifts = np.linspace(-0.02, 0.02, 31)
    sources = [brentq(lambda source: mean_at_source(source) - float(mean) - shift,
                      -0.2, 0.2, xtol=2e-15) for shift in shifts]
    coefficients = np.polynomial.Polynomial.fit(shifts, sources, 7).convert().coef
    independent3, independent4 = 2 * coefficients[2], 6 * coefficients[3]
    assert abs(independent3 - float(gamma3)) < 2e-8
    assert abs(independent4 - float(gamma4)) < 2e-6
    return dict(cumulants_exact={str(n): str(value) for n, value in
                                ((2, k2), (3, k3), (4, k4))},
                gamma3_exact=str(gamma3), gamma4_exact=str(gamma4),
                gamma3_inverse_map=independent3, gamma4_inverse_map=independent4,
                gamma3_absolute_error=abs(independent3 - float(gamma3)),
                gamma4_absolute_error=abs(independent4 - float(gamma4)))


def main():
    result = dict(scope='Dimensionless identity controls; no QICT dark-particle mass or coupling assigned.',
                  pole_sequence=pole_checks(), weak_source_visibility=weak_source_check(),
                  legendre_transform=legendre_checks(), status='PASS')
    (ROOT / 'data/PHYSICAL_IDENTITY_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: supplementary pole bounds, linear and amplitude residue enclosures, weak-source visibility, energy-origin invariance and independently inverted Legendre identities.')


if __name__ == '__main__':
    main()
