"""Numerical methods with explicit endpoint and observable conventions."""
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import kn


def correlation_sine_transform(r, amplitude, tilt, transfer):
    """xi(r) from an oscillatory quadrature on 0<k<infinity, k in Mpc^-1."""
    def amplitude_k(k):
        if k == 0:
            return 0.0
        return float(amplitude*k**(tilt+1)*transfer(k)**2)
    value, error = quad(amplitude_k, 0., np.inf, weight='sin', wvar=float(r),
                        epsabs=2e-8, limlst=250, limit=300)
    factor = 1/(2*np.pi**2*r)
    return value*factor, error*factor


def constant_rate_freezeout(coefficient, gstar=90., mass=300., x0=8., xf=4000.,
                           rtol=2e-10, max_step=.035):
    """Integrate a scaled yield and include the analytic post-xf annihilation tail.

    Radiation domination and constant gstar/rate are the defined model. The
    equilibrium contribution beyond xf is exponentially negligible at xf>=1000.
    """
    scale = 1e12
    def equilibrium(x):
        return 45/(4*np.pi**4*gstar)*x*x*kn(2,x)
    def rhs(t, u):
        x = np.exp(t)
        return [-coefficient/x/scale*(u[0]**2-(scale*equilibrium(x))**2)]
    def jac(t, u):
        return np.array([[-2*coefficient/np.exp(t)/scale*u[0]]])
    sol = solve_ivp(rhs, (np.log(x0),np.log(xf)), [scale*equilibrium(x0)],
                    method='Radau', jac=jac, rtol=rtol, atol=1e-12,
                    max_step=max_step, dense_output=True)
    if not sol.success or sol.y[0,-1] <= 0:
        raise RuntimeError(sol.message)
    endpoint = float(sol.y[0,-1]/scale)
    asymptotic = 1/(1/endpoint+coefficient/xf)
    return {'solution': sol, 'scale': scale, 'equilibrium': equilibrium,
            'Y_endpoint': endpoint, 'Y_infinity': asymptotic,
            'tail_fraction_of_endpoint': 1-asymptotic/endpoint,
            'coefficient': coefficient, 'x_endpoint': xf}
