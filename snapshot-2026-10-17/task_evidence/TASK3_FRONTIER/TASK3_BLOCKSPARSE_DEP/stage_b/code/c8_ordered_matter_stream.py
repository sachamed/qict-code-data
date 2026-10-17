#!/usr/bin/env python3
"""Exact finite L=2 ordered one-particle C8 stream for the active EW carrier.

Used as the matter-history layer for Stage B. Gauge transport is evaluated before
any averaging and the position-space order is Sz -> Sy -> Sx.  This module tests
unitarity for arbitrary link unitaries and provides deterministic route powers
for d=8,9,10; it does not perform the gauge-field PEPO contraction.
"""
from __future__ import annotations
import json, math
import numpy as np
L=2
SITES=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
SID={s:i for i,s in enumerate(SITES)}
PAULI=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.array([[1,0],[0,-1]],complex)]
I2=np.eye(2,dtype=complex)
Q=-3

def rand_su2(rng):
    q=rng.normal(size=4); q/=np.linalg.norm(q); a,b,c,d=q
    return np.array([[a+1j*d,c+1j*b],[-c+1j*b,a-1j*d]],complex)

def random_links(seed):
    rng=np.random.default_rng(seed); g={}
    for mu in range(3):
      for s in SITES:
        z=np.exp(1j*rng.uniform(-math.pi,math.pi)*Q)
        g[(mu,s)]=z*rand_su2(rng)
    return g

def shift(s,mu,sign):
    a=list(s); a[mu]=(a[mu]+sign)%L; return tuple(a)

def transporter(g,s,mu,sign):
    if sign==+1: return g[(mu,s)]
    sm=shift(s,mu,-1); return g[(mu,sm)].conj().T

def idx(site,spin,weak): return (site*2+spin)*2+weak

def stream(g,mu):
    # S_i = T_i^- P_i^+ + T_i^+ P_i^-.
    Pp=(I2+PAULI[mu])/2; Pm=(I2-PAULI[mu])/2
    M=np.zeros((len(SITES)*4,len(SITES)*4),complex)
    for si,s in enumerate(SITES):
      for sign,P in [(-1,Pp),(+1,Pm)]:
        t=shift(s,mu,sign); ti=SID[t]; U=transporter(g,s,mu,sign)
        for so in range(2):
          for ss in range(2):
            ps=P[so,ss]
            if abs(ps)==0: continue
            for wo in range(2):
              for wi in range(2):
                M[idx(ti,so,wo),idx(si,ss,wi)] += ps*U[wo,wi]
    return M

def macro(g):
    Sz=stream(g,2); Sy=stream(g,1); Sx=stream(g,0)
    return Sx@Sy@Sz, (Sx,Sy,Sz)

def self_test():
    mx=0.0; indiv=0.0; routes={}
    I=np.eye(len(SITES)*4,dtype=complex)
    for seed in range(10):
      A,ss=macro(random_links(21000+seed))
      indiv=max(indiv,*[float(np.linalg.norm(S.conj().T@S-I,2)) for S in ss])
      mx=max(mx,float(np.linalg.norm(A.conj().T@A-I,2)))
      if seed==0:
        for d in (8,9,10):
          Ad=np.linalg.matrix_power(A,d)
          routes[str(d)]={'unitarity_defect_2norm':float(np.linalg.norm(Ad.conj().T@Ad-I,2)),
                          'spectral_radius_max_abs_minus_1':float(max(abs(abs(x)-1) for x in np.linalg.eigvals(Ad)))}
    assert indiv<2e-14 and mx<3e-14
    assert max(v['unitarity_defect_2norm'] for v in routes.values())<2e-13
    return {'status':'PASS_C8_ORDERED_COMMON_PATH_STREAM',
            'geometry':{'L':2,'sites':8,'links':24},
            'carrier':{'q_U1':Q,'SU2_twice_j':1},
            'position_action_order':['Sz','Sy','Sx'],
            'random_gauge_samples':10,
            'max_substep_unitarity_defect_2norm':indiv,
            'max_macro_unitarity_defect_2norm':mx,
            'route_power_checks':routes,
            'scope':'Exact ordered matter-history engine only. Gauge plaquette/electric PEPO contraction is not performed here.'}
if __name__=='__main__': print(json.dumps(self_test(),indent=2))
