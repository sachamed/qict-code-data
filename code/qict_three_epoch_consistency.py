#!/usr/bin/env python3
"""Exact, fail-closed analytic gates for the finite-charge QICT cosmology.

No CMB likelihood, relic abundance, or cosmological evidence is inferred here.
The N^-4 scale relationship is explicitly an unproved conditional ansatz.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def coefficients(N: int):
    if not isinstance(N, int) or N < 3:
        raise ValueError("Physical charge N must be an integer >= 3")
    return (4.0 / N**2, 6.0 * (N-1) / N**2,
            (N-1)*(N-2) / N**2)


def potential_dimensionless(theta: float, N: int):
    x = 1.0 - math.cos(theta)
    a,b,c = coefficients(N)
    return a*x+b*x*x+c*x**3


def exact_coherent_expectation_binomial(theta: float, N: int):
    """Brute-force expectation of (2K/N)^3 for K~Binomial(N,x/2).
    This form is independently evaluated and is only used for N<=40.
    """
    if N > 40:
        raise ValueError("Direct binomial summation limited to N<=40")
    p = (1.0 - math.cos(theta))/2.0
    return math.fsum(
        math.comb(N,k)*(p**k)*((1-p)**(N-k))*(2.0*k/N)**3
        for k in range(N+1)
    )


def crossover(N: int):
    a,b,c = coefficients(N)
    x_3_to_2 = b/c
    x_2_to_1 = a/b
    return {
        "x_cubic_quartic_equal":x_3_to_2,
        "theta_cubic_quartic_equal_rad":math.acos(1-x_3_to_2),
        "x_quartic_quadratic_equal":x_2_to_1,
        "theta_quartic_quadratic_equal_rad":math.acos(1-x_2_to_1),
        "small_oscillation_m_phi_squared_over_V0_over_F_squared":a,
        "oscillatory_w_power6":0.5,
        "oscillatory_w_power4":1/3,
        "oscillatory_w_power2":0.0
    }


def mass_factor(theta: float, g: float):
    if g < 0: raise ValueError("g must be >=0")
    return math.sqrt(1.0+g*math.sin(theta/2.0)**2)


def beta_times_F(theta:float,g:float):
    d=1+g*math.sin(theta/2)**2
    return g*math.sin(theta)/(4*d)


def assumed_charge_vacuum_fraction(N:int,zc:float,omega_m:float,
                                    omega_r:float,omega_lam:float):
    """CONDITIONAL HYPOTHESIS rho_EDE(zc) == N**4 * rho_Lambda(0).

    This relationship is NOT derived from (1-C)^3 or the QICT receiver,
    and zc, N are input choices. Do not report as evidence.
    """
    e2=omega_m*(1+zc)**3+omega_r*(1+zc)**4
    ratio=omega_lam*N**4/e2
    return ratio/(1+ratio)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results/qict_three_epoch_consistency/analytic_checks.json")
    args=ap.parse_args()
    tol=2e-12
    # Independent finite-N checks of the coherent third-moment identity.
    ntests=0
    for n in (3,4,5,8,17,31):
        for theta in (0.,.03,.2,.7,1.7,2.5,math.pi):
            direct=exact_coherent_expectation_binomial(theta,n)
            closed=potential_dimensionless(theta,n)
            assert abs(direct-closed)<tol*max(1,abs(closed)),(n,theta,direct,closed)
            ntests+=1
    # Periodic mass and beta are tied by the exact logarithmic derivative.
    for g in (0.,.02,.1,1.):
        for t in (.03,.7,2.5):
            eps=1.e-6
            derivative=(math.log(mass_factor(t+eps,g))-
                        math.log(mass_factor(t-eps,g)))/(2*eps)
            assert abs(derivative-beta_times_F(t,g))<1.e-9,(t,g)
            ntests+=1
    n=250
    values=crossover(n)
    assert values["theta_cubic_quartic_equal_rad"]>values["theta_quartic_quadratic_equal_rad"]
    for t in (.001,.03,.1,.5):
        assert potential_dimensionless(t,n)>=0
    ntests+=5
    f=assumed_charge_vacuum_fraction(n,3500,0.315,9.3e-5,0.685)
    assert 0<f<1
    ntests+=1
    # Important: neither the early peak itself nor the N^-4 relationship
    # is a prediction of a tested gravitational microscopic Hamiltonian.
    result={
      "status":"analytic_tests_passed",
      "test_count":ntests,
      "model":"finite-N coherent potential expectation, not 1PI potential",
      "N":n,
      "coefficients":dict(zip(("x","x2","x3"),coefficients(n))),
      "crossover":values,
      "m_chi_ratio_upper_bound_g002":math.sqrt(1.02),
      "m_chi_relative_maximum_g002":math.sqrt(1.02)-1,
      "test_early_redshift":3500,
      "assumed_charge_hierarchy":{
        "status":"UNPROVEN_ANSATZ_NOT_PREDICTION",
        "postulate":"rho_EDE(zc)/rho_Lambda(0)=N**4",
        "inputs":{"Omega_m":.315,"Omega_r":9.3e-5,"Omega_Lambda":.685,"N":250,"zc":3500},
        "conditional_f_EDE_at_zc":f,
        "N_needed_for_conditional_f_0p1":(
           ((.1/.9)*(0.315*(3501**3)+9.3e-5*(3501**4))/.685)**.25
        ),
        "does_not_prove":"vacuum energy origin, Hubble tension, dark matter relic, or numerical fit"
      },
      "claims_excluded":["CMB fit", "ACT DR6","DESI DR2", "SH0ES",
                         "absolute odd-particle mass","relic abundance",
                         "cosmological constant solution"]
    }
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("QICT_THREE_EPOCH_EXACT",json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
