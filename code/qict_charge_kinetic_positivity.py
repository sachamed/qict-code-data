"""Finite-receiver convex charge inertia yields a controlled kinetic EFT.

Hamiltonian H(q)=Delta*q^2-lambda*q^4+kappa*q^6.
For Delta,lambda>0 and kappa>3lambda^2/(5Delta),
H''(q)>0 for all real q. The Legendre transform is stable and
L(v)=v^2/(4Delta)+lambda*v^4/(16Delta^4)+O(v^6).
After homogeneous canonical normalization this motivates
P(X)=X+c2 X^2+..., c2=lambda/Delta^2>0 in chosen units.
This does NOT derive a covariant relativistic continuum nor a CMB fit.
"""
import math,json
from pathlib import Path

DELTA=1.0
LAMBDA=.2
KAPPA=.5

def h(q):return DELTA*q*q-LAMBDA*q**4+KAPPA*q**6
def hp(q):return 2*DELTA*q-4*LAMBDA*q**3+6*KAPPA*q**5
def hpp(q):return 2*DELTA-12*LAMBDA*q*q+30*KAPPA*q**4

def solve_q(v):
    lo,hi=-1.0,1.0
    while hp(lo)>v:lo*=2.
    while hp(hi)<v:hi*=2.
    for _ in range(100):
        mid=(lo+hi)/2
        if hp(mid)<v:lo=mid
        else:hi=mid
    return (lo+hi)/2

def lagr(v):
    q=solve_q(v)
    return q*v-h(q)

def main():
    assert KAPPA>3*LAMBDA**2/(5*DELTA)
    minhess=2*DELTA-6*LAMBDA**2/(5*KAPPA)
    assert minhess>0
    checks=0
    for q in [-10,-5,-2,-1,0,.5,1,2,5,10]:
        assert hpp(q)>0 and h(q)>=-1e-12
        checks+=1
    cs=[]
    for v in (.0001,.001,.01,.03,.1):
        L=lagr(v)
        approx=v*v/(4*DELTA)+LAMBDA*v**4/(16*DELTA**4)
        relative=(L-approx)/(v**2/(4*DELTA))
        assert abs(relative)<.01
        cs.append({"theta_dot":v,"Legendre_L":L,"quartic_EFT_L":approx,
                   "relative_error_vs_leading_quadratic":relative})
        checks+=1
    for y in (0,.01,.1,.5,1,10):
        c2=(1+2*y)/(1+6*y)
        assert 1/3-1e-12<c2<=1+1e-12
        checks+=1
    result={
        "status":"pass",
        "tests":checks,
        "Hamiltonian":"H=Delta q²-lambda q⁴+kappa q⁶",
        "parameters":{"Delta":DELTA,"lambda":LAMBDA,"kappa":KAPPA},
        "strict_global_convexity_condition":"kappa > 3 lambda²/(5 Delta)",
        "analytical_minimum_hessian":minhess,
        "low_velocity_L":"theta_dot²/(4 Delta)+lambda theta_dot⁴/(16 Delta⁴)+O(theta_dot⁶)",
        "canonically_normalized_kinetic_EFT":"P(X)=X+(lambda/Delta²)X²+O(X³), in the chosen canonical volume units",
        "quartic_truncated_EFT_sound_speed_squared":"(1+2y)/(1+6y), y=(lambda/Delta²)*X>=0",
        "ghost_gradient_stability":"P_X>0 and P_X+2XP_XX>0 for y>=0",
        "truncated_range_cs2":"[1/3,1] for y>=0 (truncation only valid at sufficiently small y)",
        "numerical_Legendre_checks":cs,
        "missing":"derive negative quartic inertia from QICT primitive, covariant continuum, perturbation initial conditions, full Boltzmann likelihood"
    }
    p=Path("results/qict_charge_kinetic_stability/charge_kinetic_gate.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+"\n")
    print("QICT_CHARGE_KINETIC_STABILITY",json.dumps({k:v for k,v in result.items() if k!="numerical_Legendre_checks"}),flush=True)

if __name__=="__main__":main()
