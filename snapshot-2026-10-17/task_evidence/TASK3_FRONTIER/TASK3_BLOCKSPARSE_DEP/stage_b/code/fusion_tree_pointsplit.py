#!/usr/bin/env python3
"""Point-split SU(2) fusion-tree basis for the Task-3 PEPO frontier.

The six-valent cubic Gauss node is represented by a fixed binary fusion tree.
All magnetic indices are eliminated analytically.  A basis state is labelled by
six external twice-spins and the five intermediate twice-spins of a
left-associated tree ending in total spin zero.  This is an orthonormal
intertwiner basis with standard Condon-Shortley Clebsch-Gordan coefficients.

This module establishes the local representation used by the source-driven
Krylov/frontier solver.  It intentionally does not claim the global h_d values.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
from typing import Iterable, Tuple
import json, math
import numpy as np
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan


def mvals(j2:int):
    return tuple(range(-j2,j2+1,2))

def fuse(a2:int,b2:int):
    return tuple(range(abs(a2-b2),a2+b2+1,2))

@lru_cache(maxsize=None)
def cg(a2:int,ma2:int,b2:int,mb2:int,c2:int,mc2:int)->float:
    if ma2+mb2 != mc2:
        return 0.0
    return float(clebsch_gordan(Rational(a2,2),Rational(b2,2),Rational(c2,2),
                                Rational(ma2,2),Rational(mb2,2),Rational(mc2,2)))

def fusion_paths(spins2:Tuple[int,...]):
    """Return all left-associated paths ending in the singlet."""
    if len(spins2)==0:
        return [()]
    cur=[(spins2[0],())]
    for s in spins2[1:]:
        nxt=[]
        for a,p in cur:
            for c in fuse(a,s):
                nxt.append((c,p+(c,)))
        cur=nxt
    return [p for total,p in cur if total==0]

def intertwiner_amplitude(spins2,path,ms2)->complex:
    """Amplitude of a normalized chain intertwiner in the product-m basis."""
    if len(spins2)!=len(ms2):
        raise ValueError('spin/m length mismatch')
    if len(spins2)==0:
        return 1.0+0j
    if len(path)!=len(spins2)-1 or (path and path[-1]!=0):
        raise ValueError('invalid singlet path')
    J=spins2[0]; M=ms2[0]; a=1.0
    for k in range(1,len(spins2)):
        Jn=path[k-1]; Mn=M+ms2[k]
        a*=cg(J,M,spins2[k],ms2[k],Jn,Mn)
        if a==0.0:
            return 0j
        J,M=Jn,Mn
    return complex(a if M==0 else 0.0)

def dense_intertwiner(spins2,path):
    """Dense vector for validation only; never used by the global solver."""
    basis=list(product(*[mvals(j) for j in spins2]))
    v=np.array([intertwiner_amplitude(tuple(spins2),tuple(path),m) for m in basis],complex)
    return basis,v

def invariant_dimension(spins2):
    return len(fusion_paths(tuple(spins2)))

def count_binary_sixvalent(j2max:int):
    """Count local six-leg invariant blocks for external j<=jmax (diagnostic)."""
    count=0; maxmult=0; examples=[]
    for s in product(range(j2max+1), repeat=6):
        m=invariant_dimension(s)
        if m:
            count+=m; maxmult=max(maxmult,m)
            if len(examples)<5 and m>1: examples.append((s,m))
    return count,maxmult,examples

def self_test():
    # Fundamental examples.
    p2=fusion_paths((1,1,0,0,0,0)); assert len(p2)==1
    p4=fusion_paths((1,1,1,1,0,0)); assert len(p4)==2
    p6=fusion_paths((1,1,1,1,1,1)); assert len(p6)==5
    # Orthonormality at the maximal multiplicity of the j<=1/2 local sector.
    vec=[]
    for p in p6:
        _,v=dense_intertwiner((1,1,1,1,1,1),p); vec.append(v)
    G=np.array([[np.vdot(a,b) for b in vec] for a in vec])
    gram=float(np.linalg.norm(G-np.eye(len(vec))))
    assert gram<2e-14
    # Mixed-spin regression.
    spins=(2,1,1,0,2,0); ps=fusion_paths(spins)
    if ps:
        vv=[]
        for p in ps:
            _,v=dense_intertwiner(spins,p); vv.append(v)
        G2=np.array([[np.vdot(a,b) for b in vv] for a in vv])
        mixed=float(np.linalg.norm(G2-np.eye(len(vv))))
        assert mixed<2e-14
    else:
        mixed=0.0
    local_count,maxmult,examples=count_binary_sixvalent(1)
    return {
      'status':'PASS_POINT_SPLIT_FUSION_TREE_LOCAL_BASIS',
      'six_fundamental_invariant_multiplicity':len(p6),
      'six_fundamental_gram_residual':gram,
      'mixed_spin_gram_residual':mixed,
      'jhalf_or_zero_local_intertwiner_states':local_count,
      'jhalf_or_zero_max_local_multiplicity':maxmult,
      'example_multiplicity_blocks':[[list(s),m] for s,m in examples],
      'global_use':'Replace each six-valent Haar tensor by exact fusion-tree labels; magnetic indices are never stored on the PEPO frontier.',
      'scope':'Local exact basis theorem and validation only; global 24-link repeated-time contraction is separate.'
    }

if __name__=='__main__':
    print(json.dumps(self_test(),indent=2))
