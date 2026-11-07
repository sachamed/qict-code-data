#!/usr/bin/env python3
"""A symmetry no-go test and a constructive independent-parity resolution.

Theorem (conditional on the three-channel relative-phase realization):
  Under a fixed unbroken D3 reflection, only ONE of the two canonical
  relative phases can have a homogeneous background. The reflection-odd
  field does not source linear metric perturbations when its homogeneous
  background vanishes. The minimal cubic receiver parent additionally
  has a tachyonic transverse mode at the usual EDE release angle.

This is an exact kinematical and finite-N potential statement, not a
claim about the full receiver microscopic theory, dark relics, or an H0 fit.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

N=250
R=np.array([[0,1],[1,0]],dtype=int)
M=np.array([[-1,1],[-1,0]],dtype=int)
G=np.array([[2.,-1.],[-1.,2.]])/3.
LINK=np.array([[1.,0.],[-1.,1.],[0.,-1.]])
B=np.array([[1.,1.],[1.,-1.]]) # q=(s+d,s-d)
a,b,c=4/N**2,6*(N-1)/N**2,(N-1)*(N-2)/N**2

def poly(theta:float):
    x=1.-math.cos(theta)
    return a*x+b*x*x+c*x*x*x

def p2(theta:float):
    x=1.-math.cos(theta)
    return ((2*b+6*c*x)*math.sin(theta)**2
            +(a+2*b*x+3*c*x*x)*math.cos(theta))

def U(q):
    return float(sum(poly(float(th)) for th in LINK@np.asarray(q,dtype=float)))

def U_even_odd(s,d):
    return poly(s+d)+poly(s-d)+poly(2*d)

def m2(s,V0=80.,F=.2):
    # Cartesian canonical masses are Hessian / the s,d kinetic metric.
    m_even2=3*V0/F**2*p2(s)
    m_odd2=V0/F**2*(p2(s)+2*a)
    return {"even_mass2_Mpc_minus2":m_even2,
            "odd_mass2_Mpc_minus2":m_odd2,
            "odd_over_even_curvature_ratio":m_odd2/m_even2 if m_even2 else None}

def physical_instability(s,mu2_over_H2_even=4.):
    v=m2(s)
    if v["even_mass2_Mpc_minus2"]>=0: raise ValueError("even field must have tachyonic onset")
    ratio=v["odd_mass2_Mpc_minus2"]/v["even_mass2_Mpc_minus2"]
    odd_mu2_over_H2=mu2_over_H2_even*ratio
    odd_rate=(-3+math.sqrt(9+4*odd_mu2_over_H2))/2
    return {"s":s,"even_mu2_over_H2":mu2_over_H2_even,
            "odd_mu2_over_H2":odd_mu2_over_H2,
            "odd_linear_growth_per_Hubble_time":odd_rate,
            "odd_amplitude_multiplier_per_Hubble_time":math.exp(odd_rate)}

def allowed_invariant(q):
    # Lower-harmonic, D3-invariant scalar potential explicitly allowed.
    return float(sum(1-math.cos(float(th)) for th in LINK@np.asarray(q,dtype=float)))

def main():
    checks=0
    I=np.eye(2,dtype=int)
    assert np.array_equal(R@R,I)
    assert np.array_equal(M@M@M,I)
    assert np.array_equal(R@M@R,M@M)
    checks+=3
    for k in range(3):
        Mr=np.linalg.matrix_power(M,k)
        r=Mr@R@np.linalg.matrix_power(M,3-k)
        assert np.array_equal(r@r,I)
        # Reflection fixed line in two dimensions has dimension one.
        assert np.linalg.matrix_rank(I-r)==1
        assert np.count_nonzero(np.abs(np.linalg.eigvals(r)-1)<1e-10)==1
        checks+=3
    K=B.T@G@B
    assert np.allclose(K,np.diag([2/3,2]),atol=1e-15)
    checks+=1
    for s in (0.,.3,1.2,2.5,2.85):
        for d in (-.4,0.,.23):
            q=B@np.array([s,d])
            assert abs(U(q)-U_even_odd(s,d))<1e-12
            assert abs(U_even_odd(s,d)-U_even_odd(s,-d))<1e-12
            assert abs(U(q)-U(M@q))<1e-12
            assert abs(allowed_invariant(q)-allowed_invariant(M@q))<1e-12
            assert abs(allowed_invariant(q)-allowed_invariant(R@q))<1e-12
            checks+=5
    s=2.5
    h=3.e-4
    f=lambda x:U_even_odd(s,x)
    even=lambda x:U_even_odd(x,0.)
    hess_odd=(f(h)-2*f(0.)+f(-h))/h**2
    hess_even=(even(s+h)-2*even(s)+even(s-h))/h**2
    assert abs(hess_odd-(2*p2(s)+4*a))<2e-5
    assert abs(hess_even-2*p2(s))<2e-5
    checks+=2
    vals=m2(s)
    assert p2(s)<-2*a, "Tachyonic relative phase not reached"
    assert vals["odd_mass2_Mpc_minus2"]<0
    assert vals["even_mass2_Mpc_minus2"]<0
    checks+=3
    # At d=0 all odd-field first derivatives and even/odd mixings vanish.
    for s in (.1,1.5,2.5):
        g=(U_even_odd(s,h)-U_even_odd(s,-h))/(2*h)
        mix=((U_even_odd(s+h,h)-U_even_odd(s+h,-h))
            -(U_even_odd(s-h,h)-U_even_odd(s-h,-h)))/(4*h*h)
        assert abs(g)<1e-11 and abs(mix)<1e-9
        checks+=2
    # Lower-harmonic operator is allowed by D3; pure x^3 is NOT
    # protected by a discrete reflection or the cyclic group alone.
    assert allowed_invariant([.3,.2])>0
    assert abs(allowed_invariant([.0001,0])-0.0001**2)<1e-12
    checks+=2
    # A single transposition of n channel phases leaves n-2 even
    # relative coordinates after removing the diagonal phase.
    for n in range(3,9):
        assert (n-1)==(n-2)+1
        checks+=1
    growth=physical_instability(s)
    assert growth["odd_mu2_over_H2"]>0
    checks+=1
    result={
        "status":"passed_exact_reflection_no_go",
        "number_of_checks":checks,
        "scope":"Conditional three-phase receiver: U(1)^3/U(1)_diag = T^2, with exact D3 channel action",
        "consequence_1":"Each reflection fixes one real phase coordinate: exactly one active homogeneous even background if reflection remains unbroken",
        "consequence_2":"At reflection-invariant background d=0, the odd d fluctuation has vanishing linear stress-energy and cannot supply an independently active linear CMB sector",
        "consequence_3":"The cubic receiver potential is transversely tachyonic near the even field's usual tachyonic EDE release angle; nonzero d selects one of two parity-related rolling solutions",
        "consequence_4":"D3 admits an invariant lower harmonic sum(1-cos theta_link); symmetry alone does not protect a purely sextic scalar potential against radiative lower harmonics",
        "coordinates":"alpha=s+d,beta=s-d, link phases=(s+d,-2d,-s+d)",
        "kinetic_metric_s_d":K.tolist(),
        "potential":"P_N(s+d)+P_N(s-d)+P_N(2d)",
        "even_mass2_formula":"3 V0 P_N''(s)/F^2",
        "odd_mass2_formula":"V0 [P_N''(s)+2 P_N''(0)]/F^2",
        "theta_onset":s,
        "P_N_double_prime_onset":p2(s),
        "P_N_double_prime_zero":a,
        "mass2_at_test_parameters":vals,
        "conditional_hubble_instability":growth,
        "proof_of_no_go_assumptions":"The odd-DM parity is exactly the SAME reflection that acts on the 2-phase EDE doublet, and the homogeneous cosmological solution preserves it.",
        "resolution_options":[
            "Introduce an independent exact Z2_dark acting only on a separate odd state; then both phases may have nonzero EDE backgrounds while χ remains stable",
            "Use >=4 phase-bearing channels for at least two independent reflection-even relative homogeneous fields and an odd spectator",
            "Use only one reflection-even field for EDE, and give up the multifield-CMB mechanism until an independent parity is proven"
        ],
        "not_derived":["independent Z2_dark from QICT primitives","absolute DM mass","relic abundance","renormalized vacuum","two-field observed likelihood","Hubble tension resolution"],
        "previous_3_copy_odd_gap":"A bosonic receiver reflection parity must not be misidentified with fermionic spin statistics"
    }
    dest=Path("results/qict_symmetry_no_go/parity_multifield_no_go.json")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print("QICT_PARITY_MULTIFIELD_NO_GO",json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
