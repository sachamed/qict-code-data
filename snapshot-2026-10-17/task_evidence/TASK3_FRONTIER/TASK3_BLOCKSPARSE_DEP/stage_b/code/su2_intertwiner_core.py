#!/usr/bin/env python3
"""Cached SU(2) Clebsch-Gordan/intertwiner primitive for Task-3 Stage B.

This closes the algebraic recoupling primitive needed by the 24-link frontier,
without claiming that the full spacetime network has been contracted.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
import json, math
import numpy as np
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_6j

@lru_cache(maxsize=None)
def cg(j1_2:int,m1_2:int,j2_2:int,m2_2:int,J_2:int,M_2:int)->float:
    if m1_2+m2_2 != M_2: return 0.0
    args=[Rational(j1_2,2),Rational(j2_2,2),Rational(J_2,2),
          Rational(m1_2,2),Rational(m2_2,2),Rational(M_2,2)]
    return float(clebsch_gordan(*args))

def mvals(j2:int): return tuple(range(-j2,j2+1,2))

def fuse(a2:int,b2:int): return tuple(range(abs(a2-b2),a2+b2+1,2))

def singlet_paths(spins2):
    if len(spins2)<2: return [()] if spins2==(0,) else []
    states=[(spins2[0],())]
    for s in spins2[1:]:
        nxt=[]
        for a,path in states:
            for J in fuse(a,s): nxt.append((J,path+(J,)))
        states=nxt
    return [path for total,path in states if total==0]

def chain_intertwiner(spins2,path):
    """Normalized left-associated invariant vector for a path ending at J=0."""
    spins2=tuple(spins2); path=tuple(path)
    if len(path)!=len(spins2)-1 or path[-1]!=0:
        raise ValueError('path must contain n-1 intermediate twice-spins and end at zero')
    basis=list(product(*[mvals(j) for j in spins2])); v=np.zeros(len(basis),complex)
    for ix,ms in enumerate(basis):
        Jprev=spins2[0]; Mprev=ms[0]; amp=1.0
        for k in range(1,len(spins2)):
            J=path[k-1]; M=Mprev+ms[k]
            amp *= cg(Jprev,Mprev,spins2[k],ms[k],J,M)
            if amp==0.0: break
            Jprev,Mprev=J,M
        if Mprev==0: v[ix]=amp
    return basis,v

def two_spin_coupling_matrix(j1_2,j2_2):
    unc=list(product(mvals(j1_2),mvals(j2_2)))
    cou=[]
    for J in fuse(j1_2,j2_2):
      for M in mvals(J): cou.append((J,M))
    U=np.zeros((len(cou),len(unc)),float)
    for r,(J,M) in enumerate(cou):
      for c,(m1,m2) in enumerate(unc): U[r,c]=cg(j1_2,m1,j2_2,m2,J,M)
    return U

def self_test():
    U=two_spin_coupling_matrix(1,1)
    r1=float(np.linalg.norm(U@U.T-np.eye(4)))
    r2=float(np.linalg.norm(U.T@U-np.eye(4)))
    assert max(r1,r2)<1e-14
    ps=singlet_paths((1,1,1,1)); assert ps==[(0,1,0),(2,1,0)]
    vs=[]
    for p in ps:
      _,v=chain_intertwiner((1,1,1,1),p); vs.append(v)
    G=np.array([[np.vdot(a,b) for b in vs] for a in vs])
    gram=float(np.linalg.norm(G-np.eye(2)))
    assert gram<1e-14
    # Example relevant to a link carrying one gauge fundamental plus one matter fundamental.
    p2=singlet_paths((0,0,0,1,1)); assert len(p2)==1
    _,v2=chain_intertwiner((0,0,0,1,1),p2[0])
    n2=abs(float(np.vdot(v2,v2).real)-1.0); assert n2<1e-14
    # 6j symmetry/finite sanity check used by recoupling moves.
    w=float(wigner_6j(Rational(1,2),Rational(1,2),0,Rational(1,2),Rational(1,2),0))
    assert math.isfinite(w) and abs(abs(w)-0.5)<1e-14
    return {
      'status':'PASS_SU2_RECOUPLING_PRIMITIVE',
      'two_fundamental_CG_left_unitarity_residual':r1,
      'two_fundamental_CG_right_unitarity_residual':r2,
      'four_fundamental_singlet_multiplicity':len(ps),
      'four_fundamental_intertwiner_gram_residual':gram,
      'matter_link_example_norm_residual':n2,
      'sixj_example':w,
      'scope':'Exact local SU(2) recoupling primitive; does not execute the global 24-link repeated-time PEPO.'
    }

if __name__=='__main__': print(json.dumps(self_test(),indent=2))
