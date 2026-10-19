"""Evaluate and independently test the additional analytic results in the article.

All synthetic matrices below are explicitly labelled mathematical examples.
The nine-state response uses the supplied operator arrays without alteration.
Run from any directory with Python 3.12 and the pinned requirements.
"""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad, solve_ivp
from scipy.linalg import eigh, expm
from scipy.special import k1, kv
from compute_operator_closure import inverse_toeplitz_atom_upper
from numerical_methods import constant_rate_freezeout

P = Path(__file__).resolve().parents[1]


def block_moment(moments, n):
    return moments[n] if n >= 0 else moments[-n].conj().T


def toeplitz(moments, n):
    return np.block([[block_moment(moments, j-k) for k in range(n)] for j in range(n)])


def reconstruct_flat(moments, n, tolerance=1e-10):
    """Recover the cyclic unitary from an exactly flat positive moment example.

    Numerical rank selection is used only for the labelled synthetic example;
    the mathematical proposition assumes exact rank and positivity.
    """
    gram = toeplitz(moments, n)
    eigenvalues, vectors = eigh(gram)
    keep = eigenvalues > tolerance
    rank = int(keep.sum())
    assert np.linalg.matrix_rank(toeplitz(moments, n+1), tol=tolerance) == rank
    factor = np.sqrt(eigenvalues[keep])[:, None]*vectors[:, keep].conj().T
    right_inverse = vectors[:, keep]/np.sqrt(eigenvalues[keep])[None, :]
    cross = np.block([[block_moment(moments, j-k-1) for k in range(n)] for j in range(n)])
    backward = right_inverse.conj().T @ cross @ right_inverse
    source = factor[:, :moments.shape[1]]
    return backward.conj().T, source, rank


def gap_polynomial(degree, gap):
    """Chebyshev filter p_m(z), p_m(1)=1, for a certified circular gap."""
    c = np.cos(gap/2)
    z = Polynomial([0., 1.])
    previous, current = Polynomial([1.]), Polynomial([1., 1.])/(2*c)
    if degree == 0:
        return previous
    for _ in range(1, degree):
        previous, current = current, (1+z)*current/c-z*previous
    return current/np.cosh(degree*np.arccosh(1/c))


def spectral_checks():
    rng = np.random.default_rng(18102026)
    q, _ = np.linalg.qr(rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7)))
    phases = np.array([-2.7, -1.8, -.9, -.15, .65, 1.4, 2.5])
    w = q @ np.diag(np.exp(1j*phases)) @ q.conj().T
    v = np.eye(7, dtype=complex)[:, :2]
    moments = np.array([v.conj().T @ np.linalg.matrix_power(w, n) @ v for n in range(22)])
    recovered, source, rank = reconstruct_flat(moments, 4)
    error = max(np.linalg.norm(source.conj().T @ np.linalg.matrix_power(recovered, n) @ source-moments[n], 2)
                for n in range(22))
    unitarity = np.linalg.norm(recovered.conj().T @ recovered-np.eye(rank), 2)
    assert error < 2e-11 and unitarity < 1e-12

    h = rng.normal(size=(10, 10))+1j*rng.normal(size=(10, 10))
    h = (h+h.conj().T)/2
    near = expm(.02j*h)
    v = np.eye(10, dtype=complex)[:, :3]
    c1 = v.conj().T @ near @ v
    epsilon = np.linalg.norm(np.eye(3)-c1.conj().T @ c1, 2)
    ratios = []
    for n in range(2, 14):
        difference = np.linalg.norm(v.conj().T @ np.linalg.matrix_power(near, n) @ v-np.linalg.matrix_power(c1, n), 2)
        bound = min(2., n*(n-1)*epsilon/2)
        assert difference <= bound+2e-13
        ratios.append(difference/bound)

    # Noncommuting positive matrix weights test the scalar filter in Loewner order.
    theta, gap, degree = .37, .55, 12
    support = np.r_[theta, theta+np.linspace(gap, 2*np.pi-gap, 45)]
    raw = []
    for _ in support:
        b = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
        raw.append(b @ b.conj().T)
    vals, vecs = eigh(sum(raw))
    normalizer = (vecs/np.sqrt(vals)[None, :]) @ vecs.conj().T
    weights = np.array([normalizer @ b @ normalizer for b in raw])
    moments = np.array([np.einsum('i,ijk->jk', np.exp(1j*n*support), weights) for n in range(degree+1)])
    polynomial = gap_polynomial(degree, gap)
    filter_values = abs(polynomial(np.exp(-1j*(support-theta))))**2
    filtered = np.einsum('i,ijk->jk', filter_values, weights)
    a = np.concatenate([np.exp(1j*j*theta)*polynomial.coef[j]*np.eye(2) for j in range(degree+1)])
    quadratic = a.conj().T @ toeplitz(moments, degree+1) @ a
    qbound = 1/np.cosh(degree*np.arccosh(1/np.cos(gap/2)))**2
    optimized, _, _ = inverse_toeplitz_atom_upper(moments, theta)
    atom = weights[0]
    assert np.linalg.norm(filtered-quadratic, 2) < 5e-12
    assert np.linalg.eigvalsh(filtered-optimized).min() > -2e-12
    assert np.linalg.eigvalsh(qbound*(np.eye(2)-atom)-(filtered-atom)).min() > -2e-12
    assert np.linalg.eigvalsh(optimized-atom).min() > -2e-12
    return {
        'flat_reconstruction': {'example_dimension': 7, 'source_dimension': 2,
                                'rank_T4': rank, 'rank_T5': rank,
                                'max_moment_error_orders_0_to_21': float(error),
                                'unitarity_error': float(unitarity)},
        'near_invariant_source': {'epsilon': float(epsilon), 'largest_error_to_bound_ratio': float(max(ratios))},
        'gapped_filter_example': {'gap_radians': gap, 'degree': degree, 'chebyshev_remainder_factor': float(qbound),
                                 'fejer_remainder_factor': float(min(1., 1/((degree+1)**2*np.sin(gap/2)**2))),
                                 'quadratic_form_error': float(np.linalg.norm(filtered-quadratic, 2))}
    }


def energy_and_vertex_checks():
    # Same one-step unitary, different positive self-adjoint generators.
    energies = np.array([.7, 1.3])
    alias = energies+2*np.pi*np.array([1, 2])
    alias_error = float(np.max(abs(np.exp(-1j*energies)-np.exp(-1j*alias))))
    assert alias_error < 2e-15
    # Vacuum-to-one-particle sources have the same moments for every coupling.
    # The vacuum-to-two-particle propagation measures the interaction exactly.
    number = np.arange(5)
    omega, time = .8, .7
    samples = []
    for coupling in [0., .6, 1.4]:
        energies = omega*number+.5*coupling*number*(number-1)
        one = np.exp(-1j*time*energies[1])
        four = 2*np.exp(-1j*time*energies[2])
        connected = four-2*one**2
        samples.append({'coupling': coupling, 'one_particle_energy': float(energies[1]),
                        'two_particle_energy': float(energies[2]),
                        'connected_four_point_abs': float(abs(connected))})
    assert samples[0]['connected_four_point_abs'] < 1e-14
    assert samples[-1]['connected_four_point_abs'] > 1.
    return {'energy_branch_aliasing_unitary_error': alias_error, 'fixed_two_point_variable_four_point_example': samples}


def response_checks():
    with np.load(P/'code/SPECTRAL_GRAVITY_SELFTEST_INPUT.npz') as data:
        eigenvalues, vectors = eigh(data['K0'])
        ground = vectors[:, 0]
        gaps = eigenvalues[1:]-eigenvalues[0]
        a = vectors[:, 1:].conj().T @ data['K1_q'] @ ground
        b = vectors[:, 1:].conj().T @ data['K1_mq'] @ ground
        weights = abs(a)**2+abs(b)**2
    coefficients = [float(np.sum(weights/gaps**(2*n+1))) for n in range(4)]
    mu = weights/gaps/coefficients[0]
    r = 1/gaps**2
    rmin, rmax = float(r.min()), float(r.max())
    mean = coefficients[1]/coefficients[0]
    fraction = (mean-rmin)/(rmax-rmin)
    rows = []
    for s in [.01, .1, 1., 10.]:
        direct = float(np.sum(weights*gaps/(s+gaps**2)))
        lower = coefficients[0]/(1+s*mean)
        upper = coefficients[0]*((1-fraction)/(1+s*rmin)+fraction/(1+s*rmax))
        assert lower <= direct+2e-14 <= upper+4e-14
        for n in range(8):
            assert np.sum(weights*gaps/(s+gaps**2)**(n+1)) >= 0
        partial = coefficients[0]-s*coefficients[1]+s*s*coefficients[2]
        assert -1e-12 <= partial-direct <= s**3*coefficients[3]+1e-12
        rows.append({'s': s, 'response': direct, 'lower_Jensen': lower, 'upper_chord': upper})
    return {'taylor_coefficients_a0_to_a3': coefficients,
            'guaranteed_analytic_disk_radius_s': float(gaps.min()**2),
            'positive_real_axis_enclosures': rows, 'normalized_spectral_weight_sum': float(mu.sum())}


def thermal_checks():
    result = json.loads((P/'code/closure_results.json').read_text())['freezeout']
    coefficient = (2*np.pi**2/45)/1.66*np.sqrt(90)*1.22089e19*300*result['swave_GeV_minus2']
    endpoint = 30.
    initial = constant_rate_freezeout(coefficient, xf=endpoint, rtol=5e-12, max_step=.015)
    yf = initial['Y_endpoint']
    denominator = 1/yf+coefficient/endpoint
    zero_source = lambda x: 1/(1/yf+coefficient*(1/endpoint-1/x))
    equilibrium = lambda x: 45/(4*np.pi**4*90)*x*x*kv(2, x)
    integrand = lambda x: coefficient/x**2*(equilibrium(x)/zero_source(x))**2
    correction, error = quad(integrand, endpoint, np.inf, epsabs=1e-4, epsrel=1e-11)
    assert 0 <= correction < denominator
    lower, upper = 1/denominator, 1/(denominator-correction)
    # Independent continuation in log(x), with the actual nonzero source.
    scale = 1e12
    def rhs(t, y):
        x = np.exp(t)
        return [-coefficient/x/scale*(y[0]**2-(scale*equilibrium(x))**2)]
    sol = solve_ivp(rhs, (np.log(endpoint), np.log(4000.)), [scale*yf], method='DOP853',
                    rtol=3e-13, atol=1e-13, max_step=.02)
    continued = 1/(scale/sol.y[0, -1]+coefficient/4000.)
    assert lower*(1-2e-11) <= continued <= upper*(1+2e-11)
    reheat = []
    for x in [.1, 1., 10.]:
        integral = quad(lambda t: t**3*k1(t), x, np.inf, epsabs=1e-13)[0]
        sensitivity = x**4*k1(x)/integral
        step = 1e-5
        smaller = quad(lambda t: t**3*k1(t), x*np.exp(-step), np.inf, epsabs=1e-13)[0]
        larger = quad(lambda t: t**3*k1(t), x*np.exp(step), np.inf, epsabs=1e-13)[0]
        finite = (np.log(smaller)-np.log(larger))/(2*step)
        assert abs(sensitivity-finite) < 3e-8
        reheat.append({'x_reheat': x, 'dlogY_dlogT_reheat': float(sensitivity), 'finite_difference': float(finite)})
    return {'freezeout_source_tail': {'x_endpoint': endpoint, 'Y_endpoint': yf, 'Y_infinity_lower': lower,
                                      'Y_infinity_upper': upper, 'independent_continuation': continued,
                                      'relative_interval_width': upper/lower-1,
                                      'inverse_yield_correction_bound': correction,
                                      'quadrature_error_estimate': error},
            'finite_reheating_sensitivity': reheat}


def growth_and_gas_checks():
    om, radiation = .315, (2.469e-5/.674**2)*(1+.22710731766*3.046)
    lam = 1-om-radiation
    reconstructed = []
    for mu, eta in [(1., 1.), (1.05, .9)]:
        def rhs(t, y):
            a = np.exp(t)
            e2 = om/a**3+radiation/a**4+lam
            matter = om/a**3/e2
            friction = 2-(3*om/a**3+4*radiation/a**4)/(2*e2)
            return [y[1], 1.5*matter*mu*y[0]-friction*y[1]]
        p = (-.5+np.sqrt(.25+6*mu))/2
        sol = solve_ivp(rhs, (np.log(.01), 0), [.01, .01*p], dense_output=True,
                        rtol=3e-12, atol=1e-14, max_step=.01)
        t = -np.log(1.5)
        a = np.exp(t)
        f = lambda x: sol.sol(x)[1]/sol.sol(x)[0]
        step = .0005
        derivative = (f(t-2*step)-8*f(t-step)+8*f(t+step)-f(t+2*step))/(12*step)
        e2 = om/a**3+radiation/a**4+lam
        matter = om/a**3/e2
        friction = 2-(3*om/a**3+4*radiation/a**4)/(2*e2)
        inferred_mu = 2*(derivative+f(t)**2+friction*f(t))/(3*matter)
        eg = om*mu*(1+eta)/(2*f(t))
        inferred_eta = 2*f(t)*eg/(om*inferred_mu)-1
        assert abs(inferred_mu-mu) < 2e-7 and abs(inferred_eta-eta) < 4e-7
        reconstructed.append({'mu_input': mu, 'eta_input': eta, 'mu_recovered': float(inferred_mu),
                              'eta_recovered': float(inferred_eta)})
    gas = json.loads((P/'data/metric_growth_galaxy.json').read_text())['star_formation']
    outflow = gas['wind_mass_loading']/(1-gas['recycling_fraction'])*gas['stellar_mass_Msun_at_3_Gyr']
    inflow = gas['gas_inflow_Msun_per_yr']*3e9
    error = gas['gas_mass_Msun_at_3_Gyr']+gas['stellar_mass_Msun_at_3_Gyr']+outflow-inflow
    assert abs(error/inflow) < 2e-15
    return {'growth_lensing_inversion': reconstructed,
            'gas_mass_conservation': {'cumulative_wind_Msun': outflow, 'cumulative_inflow_Msun': inflow,
                                      'mass_balance_residual_Msun': error}}


def covariance_check():
    # A positive transfer measure gives a positive covariance even when TE<0.
    x = np.linspace(.01, 8., 1201)
    weight = np.exp(-x)/len(x)
    transfer = np.array([np.sin(x), -.4*np.sin(x)+.2*np.cos(2*x), np.exp(-x/2)])
    covariance = (transfer*weight) @ transfer.T
    minimum = np.linalg.eigvalsh(covariance).min()
    assert minimum > 0 and covariance[0, 1] < 0
    assert covariance[0, 1]**2 <= covariance[0, 0]*covariance[1, 1]
    # Finite filters have an exact algebraic kernel, independently of noise.
    kernels = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.], [1., 4., 9., 16.]])
    perturbation = np.array([-1., 3., -3., 1.])
    assert np.array_equal(kernels @ perturbation, np.zeros(3))
    return {'transfer_covariance_min_eigenvalue': float(minimum),
            'negative_cross_spectrum_example': float(covariance[0, 1]),
            'finite_filter_nullspace_example': perturbation.tolist()}


def main():
    result = {'spectral': spectral_checks(), 'energy_and_vertices': energy_and_vertex_checks(),
              'recorded_metric_response': response_checks(), 'thermal': thermal_checks(),
              'cosmological_inversions': growth_and_gas_checks(), 'transfer_covariance': covariance_check(),
              'all_checks_pass': True,
              'scope': 'Analytic propositions and independent numerical examples in their stated domains; physical operator inputs retain their explicit provenance.'}
    (P/'data/ANALYTIC_EXTENSION_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
