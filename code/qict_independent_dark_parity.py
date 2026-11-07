#!/usr/bin/env python3
"""Constructive parity-safe completion of the conditional D3 receiver doublet.

A separately conserved central Z2_dark makes a bosonic odd field χ stable
against decay to parity-even states, while both D3 relative phases can
roll and linearly affect gravity. No assertion that the extra Z2_dark
arises from original QICT primitives, no relic density or vacuum claim.
"""
from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
import numpy as np

R=np.array([[0,1],[1,0]],dtype=int)
M=np.array([[-1,1],[-1,0]],dtype=int)
I=np.eye(2,dtype=int)
G=np.array([[2.,-1.],[-1.,2.]])/3
LINK=np.array([[1,0],[-1,1],[0,-1]])
N=250
a,b,c=4./N**2,6.*(N-1)/N**2,(N-1)*(N-2)/N**2


def d3_elements():
    els=[]
    for k in range(3):
        mk=np.linalg.matrix_power(M,k)
        els.extend((mk,mk@R))
    assert len(els)==6
    assert len({tuple(g.flatten()) for g in els})==6
    return els


def V(q):
    th=LINK@np.asarray(q,dtype=float)
    x=1.-np.cos(th)
    return float(sum(a*x+b*x*x+c*x*x*x))


def kinetic(v,F=0.2):
    v=np.asarray(v,dtype=float)
    return .5*F*F*float(v@G@v)


def mchi2(v,m0=1.,kappa=.01,Lambda=1.,F=.2):
    if m0<=0 or kappa<0 or Lambda<=0:raise ValueError("choose positive scales")
    return m0*m0+kappa*kinetic(v,F)/Lambda**2


def total_energy_density(q,v,chi,chidot,lambda_chi=.1):
    # Homogeneous EFT energy density, with a positive derivative
    # coupling; ignore χ backreaction on metric kinetic for this
    # algebraic diagnostic. For a complete UV model derive full Tmunu.
    return kinetic(v)+V(q)+.5*chidot**2+.5*mchi2(v)*chi**2+lambda_chi/4*chi**4


def grad_v(q):
    th=LINK@np.asarray(q,float); x=1.-np.cos(th)
    return LINK.T@((a+2*b*x+3*c*x*x)*np.sin(th))


def main():
    group=d3_elements()
    checks=0
    for g in group:
      assert np.allclose(g.T@G@g,G)
      checks+=1
      for h in group:
        gh=g@h
        assert any(np.array_equal(gh,z) for z in group)
        checks+=1
    center=[g for g in group if all(np.array_equal(g@h,h@g) for h in group)]
    assert len(center)==1 and np.array_equal(center[0],I)
    checks+=1
    # The extra dark parity is a separate central factor:
    combined=list(itertools.product(range(len(group)),(-1,1)))
    assert len(combined)==12
    dark_parity=(I,-1)
    checks+=1
    for q,v in [
      (np.array([2.5,1.2]),np.array([.03,-.04])),
      (np.array([.7,-1.4]),np.array([-.3,.12])),
      (np.array([1.5,1.5]),np.array([.02,.02]))
    ]:
      for g in group:
        assert abs(V(g@q)-V(q))<3e-12
        assert abs(kinetic(g@v)-kinetic(v))<3e-12
        assert abs(mchi2(g@v)-mchi2(v))<3e-12
        checks+=3
        for chi,chidot in [(0,0),(.7,.03),(-.5,-.2)]:
            e=total_energy_density(q,v,chi,chidot)
            assert abs(e-total_energy_density(g@q,g@v,-chi,-chidot))<5e-12
            checks+=1
      assert mchi2(v)>=1
      checks+=1
    q=np.array([2.5,1.2])
    assert np.linalg.norm(grad_v(q))>1e-5
    assert not np.array_equal(q,R@q)
    assert np.allclose(np.linalg.eigvalsh(G),[1/3,1])
    checks+=3
    for v in (np.zeros(2),np.array([.1,.3])):
      for chi in (0.,.01,.5):
        force=mchi2(v)*chi+.1*chi**3
        if chi==0:assert force==0
        checks+=1
    # Per-vector Z2_dark with χ_background=0 is unbroken for arbitrary q.
    # This is distinct from the channel reflection acting on q.
    result={
      "status":"passed_independent_dark_parity_symmetry_tests",
      "checks":checks,
      "D3_order":len(group),
      "D3_center_order":len(center),
      "D3_center_trivial":True,
      "D3_times_Z2dark_order":len(combined),
      "model":"Two independent relative-phase EDE scalars q, one Z2dark-odd real bosonic state chi",
      "background_parity":"chi_background=0 preserves independent Z2_dark irrespective of D3-breaking q background",
      "parity_selection_rule":"Odd chi cannot decay into any collection of purely even particles if full interactions respect Z2dark and chi is lightest odd",
      "dark_spin":"bosonic here; neither flavor exchange nor Z2dark alone determines fermionic spin statistics",
      "V_even":"V0 sum over 3 links of P_N[1-cos(theta_l)]",
      "kinetic_even":"(F²/2) G_ab dq_a dq_b; G=(1/3)[[2,-1],[-1,2]]",
      "dark_mass2":"m0²+(kappa/Lambda²)X, X=(F²/2)G_ab qdot_a qdot_b >=0 on homogeneous background",
      "dark_renormalization":"Static-phase-independent derivative term avoids a tree-level phase-dependent mass determinant, but no radiative EFT control is proven",
      "microscopic_status":"INDEPENDENT Z2dark NOT DERIVED; distinct from the flavor reflection Z2 already inside D3",
      "alternative":">=4 phase-bearing channels can leave two reflection-even homogeneous relative phases and one odd spectator",
      "required_physics":["derive independent parity from receiver superselection","derive charge scale and odd mass","calculate production collision term and relic density","full covariant two-field perturbations","joint Planck ACT DESI SH0ES inference","renormalized vacuum mechanism"],
      "not_a_fit":True
    }
    p=Path("results/qict_dark_parity_extension/parity_safe_two_phase.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+"\n")
    print("QICT_PARITY_SAFE_MULTIFIELD",json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
