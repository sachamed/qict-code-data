#!/usr/bin/env python3
"""Source-authentic A7 Kato ground and relative GNS moments on one L24 plaquette.

The orientation comes from the genuine L24 payload.  The state rule K_E+K_B
is the existing A7 rule in supplement s09.  U0 and Uq share the same finite
flux regulator. The six insertions are Wilson loop/string probes in external
static charge sectors, not CAR chiral/family sources or physical C10 closure.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.fft import dst
from scipy.linalg import eigh_tridiagonal

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reproducibility/data/common_physical_chain"
OUT = ROOT / "reproducibility/results/common_primitive_kato"
NMAX = 12
POLYNOMIAL_ORDER = 14
PHI = math.pi/63
ACTION = ROOT / "reproducibility/data/physical_g2_inputs/L24_RANK6_STATE_CONTRACTION_PAYLOAD_2026-10-04/components/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json"


def rational_upper_sqrt(x: Fraction) -> Fraction:
    with localcontext() as ctx:
        ctx.prec = 85
        y = (Decimal(x.numerator)/Decimal(x.denominator)).sqrt()
        upper = Fraction(y)
        if upper*upper < x:
            upper += Fraction(1, 10**(ctx.prec-10))
        assert upper*upper >= x
    return upper


def rational_state_certificate(v: np.ndarray, flux: np.ndarray) -> dict:
    # Exact rational witness from serialized decimal vector components.
    x = [Fraction(format(float(z), '.17g')) for z in v]
    norm2 = sum(z*z for z in x)
    hv = [(8*int(n)*int(n)+4)*x[i]
          - (2*x[i-1] if i else 0)
          - (2*x[i+1] if i+1 < len(x) else 0)
          for i, n in enumerate(flux)]
    mu = sum(a*b for a,b in zip(x,hv))/norm2
    r2 = (sum((b-mu*a)**2 for a,b in zip(x,hv))+4*x[0]**2+4*x[-1]**2)/norm2
    residual = rational_upper_sqrt(r2)
    gap = 8-mu
    assert 0 < mu < 4 and gap > 4
    td = residual/gap
    zero_index = int(np.flatnonzero(flux == 0)[0])
    convolution_margin = abs(x[zero_index])-sum(abs(a) for i,a in enumerate(x) if i != zero_index)
    assert convolution_margin > 0
    gram_lower = convolution_margin**2/norm2
    assert gram_lower > Fraction(24,100)
    return {"rayleigh_rational": str(mu), "rayleigh_value": float(mu),
            "infinite_operator_residual_squared_rational": str(r2),
            "residual_upper_rational": str(residual),
            "residual_upper": float(residual),
            "ground_energy_lower": float(mu-residual),
            "ground_energy_upper": float(mu),
            "gap_lower": float(gap),
            "ground_state_trace_distance_upper": float(td),
            "state_vector_distance_upper": float(math.sqrt(2)*td),
            "source_gram_lower_rational": str(gram_lower),
            "source_gram_lower": float(gram_lower),
            "argument": "H>=8r^2, E1>=8, Rayleigh ground bound <=mu<4. "
                        "Complement overlap <=residual/(8-mu); PF identifies the simple ground."}


def magnetic_action(x: np.ndarray) -> np.ndarray:
    # Exact unitary finite flux regulator via orthonormal DST-I diagonalization.
    dimension = x.shape[0]
    k = np.arange(1, dimension+1)
    bphase = np.exp(-1j*PHI*(4-4*np.cos(np.pi*k/(dimension+1))))
    y = dst(x, type=1, axis=0, norm='ortho')
    y *= bphase if y.ndim == 1 else bphase[:, None]
    return dst(y, type=1, axis=0, norm='ortho')


def update(x: np.ndarray, flux: np.ndarray, q: int) -> np.ndarray:
    cost = 8*flux*flux+4*q*flux+2*q*q+q
    ephase = np.exp(-1j*np.pi*((cost % 126)/63))
    return magnetic_action(ephase*x if x.ndim == 1 else ephase[:, None]*x)


def update_adjoint(x: np.ndarray, flux: np.ndarray, q: int) -> np.ndarray:
    # Uq^dagger=E_q^dagger B_D^dagger: the order is reversed explicitly.
    y = np.conj(magnetic_action(np.conj(x)))
    cost = 8*flux*flux+4*q*flux+2*q*q+q
    phase = np.exp(1j*np.pi*((cost % 126)/63))
    return phase*y if y.ndim == 1 else phase[:, None]*y


def direct_hilbert_schmidt_check(v: np.ndarray, probes: np.ndarray,
                                flux: np.ndarray, q: int,
                                moments: np.ndarray) -> float:
    # Independent dense HS propagation for only two steps on the smallest
    # regulator; this retains the full neutral right action rather than using
    # the factorized return-amplitude formula being checked.
    errors = []
    for j in range(6):
        y = np.outer(probes[:, j], v.conj())
        for n in range(1, 3):
            y = update(update(y.conj().T, flux, 0).conj().T, flux, q)
            column = probes.conj().T @ (y @ v)
            errors.append(float(np.linalg.norm(column-moments[n, :, j])))
    error = max(errors)
    assert error < 1e-12
    return error


def normalized_probes(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.zeros((len(v), 6), complex)
    for j in range(6):
        raw[j:, j] = v[:len(v)-j] if j else v
    gram = raw.conj().T @ raw
    ev, u = np.linalg.eigh(gram)
    assert ev.min() > .01
    invsqrt = (u*ev**-.5) @ u.conj().T
    probes = raw @ invsqrt
    assert np.linalg.norm(probes.conj().T@probes-np.eye(6)) < 2e-14
    return probes, gram


def relative_moments(v: np.ndarray, probes: np.ndarray,
                     flux: np.ndarray, q: int) -> tuple[np.ndarray, dict]:
    left, right = probes.copy(), v.copy()
    moments = []
    normerr = 0.
    for n in range(NMAX+1):
        # vec(Uq^n |b><v| U0^-n): both factors use this identical grid.
        a0 = np.vdot(v, right)
        moments.append(np.conj(a0)*(probes.conj().T@left))
        normerr = max(normerr, float(abs(np.linalg.norm(right)-1)),
                      float(np.linalg.norm(left.conj().T@left-np.eye(6))))
        left = update(left, flux, q)
        right = update(right, flux, 0)
    c = np.asarray(moments)
    # Independently propagate negative powers using the reversed factor order.
    left, right = probes.copy(), v.copy()
    adjoint_error = 0.
    inverse_error = float(np.linalg.norm(update_adjoint(update(v, flux, q), flux, q)-v))
    for n in range(NMAX+1):
        negative = np.conj(np.vdot(v, right))*(probes.conj().T@left)
        adjoint_error = max(adjoint_error, float(np.linalg.norm(negative-c[n].conj().T)))
        left = update_adjoint(left, flux, q)
        right = update_adjoint(right, flux, 0)
    assert adjoint_error < 1e-12 and inverse_error < 1e-12
    t = np.block([[c[j-i] if j >= i else c[i-j].conj().T
                   for j in range(NMAX+1)] for i in range(NMAX+1)])
    return c, {"C0_identity_frobenius_error": float(np.linalg.norm(c[0]-np.eye(6))),
               "unitary_orbit_norm_error": normerr,
               "independent_negative_moment_adjoint_frobenius_error": adjoint_error,
               "ordered_update_inverse_norm_error": inverse_error,
               "toeplitz_smallest_eigenvalue_diagnostic": float(np.linalg.eigvalsh(t).min()),
               "nonstationary_neutral_state_retained": True,
               "positive_exact_toeplitz_by_relative_unitary_gram": True}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    action = json.loads(ACTION.read_text())
    assert action['phi_star'] == 'pi/63'
    assert action['electric_generator']['U1'] == '2 E_Y^2 + E_Y'
    assert action['plaquette_generator']['U1'] == '4 [1-cos(theta_p)]'
    payload = DATA / "L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz"
    with np.load(payload, allow_pickle=False) as z:
        eta = np.asarray(z['plaquette_edge_sign'][0], int)
        edges = np.asarray(z['plaquette_edge_indices'][0], int)
    assert eta.tolist() == [1, 1, -1, -1]
    assert 2*np.sum(eta*eta) == 8 and np.sum(eta) == 0
    # Magnetic Taylor polynomial locality: each degree moves one flux unit.
    # Conservative x=0.4 > 8*pi/63, giving an exact rational exponential-tail bound.
    x = Fraction(2, 5)
    first = x**(POLYNOMIAL_ORDER+1)/math.factorial(POLYNOMIAL_ORDER+1)
    # e^.4<1.5, a simple series bound (sum to 8 plus geometric remainder).
    e_upper = sum(x**j/math.factorial(j) for j in range(9))
    e_upper += (x**9/math.factorial(9))/(1-x/Fraction(10))
    assert e_upper < Fraction(3, 2)
    tail = e_upper*first
    relative_boundary_bound = 4*NMAX*tail*(1+tail)**(2*NMAX-1)
    reports, arrays = [], {}
    prev = {}
    for radius in [8, 12, 16, 20]:
        sflux = np.arange(-radius, radius+1)
        energies, vecs = eigh_tridiagonal(8*sflux*sflux+4,
                                         np.full(len(sflux)-1, -2.),
                                         select='i', select_range=(0, 1))
        ground = vecs[:, 0]
        if ground[np.argmax(abs(ground))] < 0: ground = -ground
        ground = ground/np.linalg.norm(ground)
        state_cert = rational_state_certificate(ground, sflux)
        margin = NMAX*POLYNOMIAL_ORDER+2
        flux = np.arange(-radius-margin, radius+5+margin+1)
        v = np.zeros(len(flux), complex)
        v[(sflux-flux[0]).astype(int)] = ground
        probes, gram = normalized_probes(v)
        neutral_return = complex(np.vdot(v, update(v, flux, 0)))
        neutral_stationarity_distance = math.sqrt(max(0., 1-abs(neutral_return)**2))
        gmin = float(np.linalg.eigvalsh(gram).min())
        dv = state_cert['state_vector_distance_upper']
        di = math.sqrt(6)*dv
        dg = 12*dv
        # Independent exact convolution bound exceeds .24; the perturbation is
        # below .01, so g=.23 is a conservative certified denominator.
        gm = .23
        assert dg < .01
        # Norm bound for positive Gram normalization, then the neutral right leg.
        source_error = di/math.sqrt(gm)+math.sqrt(6)*dg/(2*gm**1.5)+dv
        total_error = 2*source_error+float(relative_boundary_bound)
        qrecords = {}
        for q in (-3, -6):
            c, diag = relative_moments(v, probes, flux, q)
            if radius == 8:
                diag['independent_direct_HS_two_step_error'] = direct_hilbert_schmidt_check(v, probes, flux, q, c)
            change = (float(max(np.linalg.norm(a-b, 2) for a,b in zip(c, prev[q])))
                      if q in prev else None)
            prev[q] = c
            qrecords[str(q)] = {**diag,
                'successive_state_cutoff_moment_difference_max': change,
                'analytic_state_and_finite_flux_boundary_error_upper': total_error}
            arrays[f'R{radius}_q{q}_moments'] = c
        arrays[f'R{radius}_common_flux_grid'] = flux
        arrays[f'R{radius}_rho0_gram_factor'] = v[:, None]
        arrays[f'R{radius}_source_gram'] = gram
        arrays[f'R{radius}_left_probe_isometry'] = probes
        reports.append({'primitive_state_flux_cutoff': radius,
                        'common_evolution_flux_dimension': len(flux),
                        'Kato_generator': 'K_prim=8r^2+4-2(S+S^dagger)',
                        'Kato_reference_gap_uniform_lower': 4,
                        'ground_and_gap_certificate': state_cert,
                        'primitive_density_stationarity_trace_distance_diagnostic': neutral_stationarity_distance,
                        'source_gram_min_eigenvalue': gmin,
                        'static_charge_sector_results': qrecords})
    npz = OUT / 'COMMON_A7_KATO_STATIC_GNS_MOMENTS.npz'
    np.savez_compressed(npz, **arrays)
    result = {
        'status': 'SOURCE_A7_COMMON_REGULATOR_KATO_AND_RELATIVE_GNS_CERTIFIED_ON_ONE_PLAQUETTE',
        'scope': 'Gauge-only authentic L24-oriented plaquette, external static charge sectors. '
                 'Six Wilson probes are not physical CAR chirality/family sources.',
        'payload_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(),
        'frozen_action_sha256': hashlib.sha256(ACTION.read_bytes()).hexdigest(),
        'actual_first_plaquette_edges': edges.tolist(),
        'actual_orientation_coefficients': eta.tolist(),
        'neutral_electric_cost': '8r^2 (the signed linear terms cancel)',
        'charged_external_electric_cost': '8r^2+4qr+2q^2+q',
        'state_rule_source': 'paper/sections/supplement/s09.tex primitive A7 cost K_E+K_B',
        'six_insertions': 'D_q times closed-loop flux shifts 0,1,2,3,4,5',
        'GNS_transfer': 'conjugate(U0) tensor Uq on HS(H0,Hq)',
        'moment_formula': 'C_n=conjugate(<v|U0^n|v>)*B^dagger Uq^n B',
        'full_C10_chiral_or_Kato_endpoint_certified': False,
        'finite_flux_boundary_polynomial_order': POLYNOMIAL_ORDER,
        'analytic_relative_moment_boundary_error_upper': float(relative_boundary_bound),
        'cutoff_results': reports,
        'generated_data_sha256': hashlib.sha256(npz.read_bytes()).hexdigest(),
        'numerical_policy': 'Rational certificates bound state and flux truncation. '
                            'DST/eigendecomposition roundoff and diagnostic comparisons '
                            'are separately reported at floating-point precision.',
        'total_floating_point_moment_error_interval_certified': False,
    }
    (OUT / 'COMMON_A7_KATO_STATIC_GNS_CERTIFICATE.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'ground_energy': reports[-1]['ground_and_gap_certificate']['rayleigh_value'],
                      'ground_gap_lower': reports[-1]['ground_and_gap_certificate']['gap_lower'],
                      'four_source_gram_min': [r['source_gram_min_eigenvalue'] for r in reports],
                      'boundary_error_upper': float(relative_boundary_bound),
                      'full_physical_C10': False}, indent=2))


if __name__ == '__main__':
    main()
