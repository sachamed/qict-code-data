"""Covariant one-pole background and conserved variable-mass cold particle checks.

A minimal public verification kernel; the journal source archive contains the
full Boltzmann adapter, thermal neutrino integrals and original test suite.
Only homogeneous and algebraic identities are evaluated here.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

CRITICAL_LAMBDA = 1.816666769680014

def receiver_fraction(E, lam=0.03776, scale=1e5):
    x = np.asarray(E) / scale
    return lam * x / (1 + x*x)

def positive_branch(a, lam=0.03776, scale=1e5, om=0.2945381129026236,
                    rad=8.630e-5):
    """Unique expanding E=H/H0 for massless-neutrino illustrative background."""
    if not 0 <= lam < CRITICAL_LAMBDA:
        raise ValueError("Friedmann monotonicity condition violated")
    a = np.asarray(a, dtype=float)
    floor = 1 - om - rad - receiver_fraction(1.,lam,scale)
    if floor <= 0 or np.any(a<=0): raise ValueError("inadmissible background")
    rhs = om*a**-3 + rad*a**-4 + floor
    E = np.array([brentq(lambda x:x*x*(1-receiver_fraction(x,lam,scale))-r,
                         np.sqrt(r)*(1-1e-10),
                         np.sqrt(r)/np.sqrt(1-lam/2)*(1+1e-10))
                  for r in np.atleast_1d(rhs)])
    return E, rhs, floor

def beta(x,g):
    return g*g*x/(1+g*g*x*x)

def beta_gradient(x,g):
    return g*g*(1-g*g*x*x)/(1+g*g*x*x)**2

def derivative(N, state, *, mass_ratio=0.8, ob=.05, og=5e-5,
               potential=lambda x: (.7+.015*x*x, .03*x)):
    """State x=phi/Mpl, v=dx/dN, rc=rho_c/(3 Mpl^2 H0^2)."""
    x,v,rc=state
    a=np.exp(N);rb=ob*a**-3;rg=og*a**-4
    U,Ux=potential(x)
    E2=(rb+rg+rc+U)/(1-v*v/6)
    hN=-(3*(rb+rc)+4*rg+E2*v*v)/(2*E2)
    b=beta(x,mass_ratio)
    return [v,-(3+hN)*v-3*(Ux+b*rc)/E2,(-3+b*v)*rc]

def evolve(N0=-2,N1=0,g=.8):
    rc0=.25*np.exp(-3*N0)
    sol=solve_ivp(lambda N,y:derivative(N,y,mass_ratio=g),
                  (N0,N1),[.2,.012,rc0],rtol=1e-10,atol=1e-12,
                  t_eval=np.linspace(N0,N1,250))
    if not sol.success:raise RuntimeError(sol.message)
    return sol

def cold_linear_sources(a, Hc, phip, dphi, dphip, Psi, Phi_p,
                        Psi_p, Vphi, Vphiphi, rho_c, dc, theta, k, beta0, beta_p):
    """Newtonian gauge; primed derivatives are in conformal time."""
    dc_p=-theta+3*Phi_p+beta0*dphip+beta_p*phip*dphi
    theta_p=-(Hc+beta0*phip)*theta+k*k*(Psi+beta0*dphi)
    kg_rhs=phip*(Psi_p+3*Phi_p)-2*a*a*(Vphi+beta0*rho_c)*Psi-a*a*beta0*rho_c*dc
    kg_mass2=Vphiphi+beta_p*rho_c
    return dc_p,theta_p,kg_rhs,kg_mass2
