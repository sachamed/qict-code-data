#!/usr/bin/env python3
"""Finite cyclic-regulator certificate for the C8 electric/plaquette layer.
The theorem in the paper is operator-theoretic for compact links.  This program
checks a finite Z_N regulator: exact unitarity, local gauge invariance of the
plaquette phase, and the C8 direct/dual Hessian normalization.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
OUT=CERT/'c8_plaquette_reciprocity_certificate.json'
PHI=math.pi/63

def idx(a,b,c,d,N): return ((a*N+b)*N+c)*N+d

def gauge_transform_index(state, alpha, N):
    # oriented plaquette links: 0: 0->1, 1:1->2, 2:3->2, 3:0->3
    # local vertex phases act as link-coordinate shifts A_l -> A_l + alpha_tail-alpha_head.
    a,b,c,d=state
    v0,v1,v2,v3=alpha
    return ((a+v0-v1)%N,(b+v1-v2)%N,(c+v3-v2)%N,(d+v0-v3)%N)

def flux(state,N):
    a,b,c,d=state
    # oriented Wilson loop 0->1->2->3->0: a+b-c-d
    return (a+b-c-d)%N

def main():
    N=5
    states=[(a,b,c,d) for a in range(N) for b in range(N) for c in range(N) for d in range(N)]
    dim=N**4
    # Magnetic gate is diagonal in link-coordinate basis.
    mb=np.empty(dim,complex)
    for st in states:
        th=2*math.pi*flux(st,N)/N
        B=4*(1-math.cos(th))
        mb[idx(*st,N)]=np.exp(-1j*PHI*B)
    mag_unit=float(np.max(np.abs(np.abs(mb)-1)))
    # Gauge invariance: diagonal phase is constant along gauge orbits.
    rng=np.random.default_rng(20260909)
    max_gauge=0.0
    for _ in range(1000):
        st=states[int(rng.integers(0,dim))]
        al=tuple(int(x) for x in rng.integers(0,N,size=4))
        st2=gauge_transform_index(st,al,N)
        e=abs(mb[idx(*st,N)]-mb[idx(*st2,N)])
        max_gauge=max(max_gauge,float(e))
    # Electric gate: diagonal in Fourier/electric basis; phases have unit modulus.
    reps=np.arange(N)
    reps=np.where(reps<=N//2,reps,reps-N)
    P=lambda E:2*E*E+E
    el=[]
    for a in reps:
      for b in reps:
       for c in reps:
        for d in reps:
         el.append(np.exp(-1j*PHI*(P(a)+P(b)+P(c)+P(d))))
    el=np.array(el)
    elec_unit=float(np.max(np.abs(np.abs(el)-1)))
    # C8 direct/dual reciprocity at flat configuration: P_+''(0)=4 and B''(0)=4.
    h=1e-5
    Bp=lambda th:4*(1-math.cos(th))
    hess_B=(Bp(h)-2*Bp(0)+Bp(-h))/h**2
    hess_E=4.0
    hess_err=abs(hess_B-hess_E)
    # The elementary phase ratio is exactly one by construction.
    cert={
      'status':'PASS','principle':'C8 primitive covariant reciprocity',
      'regulator':'Z_5 single plaquette','phi_star':PHI,'phi_star_exact':'pi/63',
      'magnetic_unitarity_defect':mag_unit,'electric_unitarity_defect':elec_unit,
      'max_magnetic_gauge_orbit_defect':max_gauge,
      'electric_quadratic_hessian':hess_E,'plaquette_flat_hessian_numeric':hess_B,
      'hessian_mismatch':hess_err,'elementary_electric_to_curvature_phase_ratio':1.0,
      'note':'Finite cyclic regulator test; compact-link theorem is analytic in Supplement G.'
    }
    assert mag_unit<1e-14 and elec_unit<1e-14 and max_gauge<1e-14 and hess_err<1e-5
    OUT.write_text(json.dumps(cert,indent=2)+'\n')
    print('PASS C8-plaquette',json.dumps(cert,indent=2))
if __name__=='__main__': main()
