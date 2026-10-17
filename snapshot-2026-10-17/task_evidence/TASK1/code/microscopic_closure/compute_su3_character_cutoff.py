#!/usr/bin/env python3
"""Deterministic SU(3) plaquette character expansion and rigorous Taylor-tail cutoff.

For the frozen color plaquette gate with x=A3*pi/63=4*pi/7,
f(U)=exp[-i x (1-Re Tr_3 U/3)].  The exact character expansion coefficients
are checked by Weyl integration.  A representation cutoff max(p,q)<=R is
rigorously controlled by the total-order exponential tail because an irrep
with max(p,q)>R cannot occur in tensor order <=R of 3 and bar{3}.
"""
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'global_sm_extension'/'SU3_CHARACTER_CUTOFF.json'; OUT.parent.mkdir(parents=True,exist_ok=True)
x=4*math.pi/7
GATES=240
TARGET=1e-6

def dim(p,q): return (p+1)*(q+1)*(p+q+2)//2

def exp_tail(R):
    return math.exp(x)-sum(x**n/math.factorial(n) for n in range(R+1))
R=0
while GATES*exp_tail(R)>TARGET/4: R+=1
# Reserve <=25% of total error for SU3 character truncation.

def det3(a,b,c,d,e,f,g,h,i): return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)

def quadrature(M=256,pmax=10):
    th=(np.arange(M)+0.5)*2*np.pi/M-np.pi
    t1,t2=np.meshgrid(th,th,indexing='ij')
    z1=np.exp(1j*t1); z2=np.exp(1j*t2); z3=np.exp(-1j*(t1+t2)); one=np.ones_like(z1)
    D=det3(z1**2,z1,one,z2**2,z2,one,z3**2,z3,one)
    fval=np.exp(-1j*x*(1-np.real(z1+z2+z3)/3))
    coeff={}
    for p in range(pmax+1):
      for q in range(pmax+1):
        N=det3(z1**(p+q+2),z1**(q+1),one,
               z2**(p+q+2),z2**(q+1),one,
               z3**(p+q+2),z3**(q+1),one)
        coeff[(p,q)]=np.mean(np.conj(N)*D*fval)/6
    shells=[]
    for r in range(pmax+1):
        shells.append(float(sum(abs(c)*dim(p,q) for (p,q),c in coeff.items() if max(p,q)==r)))
    return coeff,shells
c128,s128=quadrature(128,10); c256,s256=quadrature(256,10)
conv=max(abs(c128[k]-c256[k]) for k in c128)
obj={
 'status':'EXECUTED_SU3_LOCAL_CHARACTER_EXPANSION',
 'frozen_input':{'A3':36,'phi_star':'pi/63','x=A3*phi_star':'4*pi/7','x_numeric':x,'gate_count_for_n_le_10':GATES},
 'rigorous_tail_argument':'f=e^{-ix} exp[(ix/6)(chi_3+chi_bar3)]. An irrep with max(p,q)>R cannot occur at tensor order <=R. Since |chi_3|,|chi_bar3|<=3, the omitted sup norm is bounded by sum_{n>R} x^n/n!.',
 'chosen_representation_cutoff':{'max_p_q':R,'single_gate_sup_bound':exp_tail(R),'240_gate_union_bound':GATES*exp_tail(R),'allocated_budget':TARGET/4},
 'dense_Peter_Weyl_dimension_at_cutoff':sum(dim(p,q)**2 for p in range(R+1) for q in range(R+1)),
 'number_of_irrep_blocks_at_cutoff':(R+1)**2,
 'deterministic_Weyl_quadrature_crosscheck':{
   'M_values':[128,256],
   'max_coefficient_change_pq_le_10':float(conv),
   'c00_M256':[float(c256[(0,0)].real),float(c256[(0,0)].imag)],
   'c10_M256':[float(c256[(1,0)].real),float(c256[(1,0)].imag)],
   'dimension_weighted_shells_max_pq_0_to_10':s256,
 },
 'consequence':'Color is computationally compatible with the same symmetry-blocked contraction, but a dense Peter-Weyl link basis is prohibitive. The SU(3) extension must remain block-sparse in (p,q) and color-intertwiner multiplicity.'
}
OUT.write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
