#!/usr/bin/env python3
"""Exact SU(2) plaquette-corner insertion in a six-leaf fusion-tree basis.

A plaquette representation r entering a cubic vertex on one incident link and
leaving on a second is contracted with the SU(2) invariant epsilon tensor.  The
matrix element is evaluated between normalized six-leg Gauss intertwiners.  The
link Peter-Weyl normalization sqrt(d_j/d_J) is deliberately kept outside this
vertex coefficient; four corners plus four link factors reproduce unit
amplitude when a single character Wilson loop acts on the vacuum.

The implementation below uses explicit magnetic sums only as a local reference
and table generator.  It is not used as a global dense Hilbert-space method.
"""
from __future__ import annotations
from itertools import product
from functools import lru_cache
import json, math
import numpy as np
from fusion_tree_pointsplit import mvals, fuse, fusion_paths, intertwiner_amplitude, cg


def eps_phase(r2:int,a2:int)->int:
    # (-1)^(r-a), integer exponent in twice-spin notation.
    return -1 if (((r2-a2)//2)&1) else 1


def corner_element(old_spins,old_path,new_spins,new_path,leaf_i,leaf_j,r2):
    """<I_new|G_r(i,j)|I_old>, without link sqrt(d_old/d_new) factors."""
    old_spins=tuple(old_spins);new_spins=tuple(new_spins)
    if len(old_spins)!=6 or len(new_spins)!=6 or leaf_i==leaf_j:
        raise ValueError('six leaves and distinct corner leaves required')
    untouched=[k for k in range(6) if k not in (leaf_i,leaf_j)]
    if any(old_spins[k]!=new_spins[k] for k in untouched): return 0j
    if new_spins[leaf_i] not in fuse(old_spins[leaf_i],r2): return 0j
    if new_spins[leaf_j] not in fuse(old_spins[leaf_j],r2): return 0j
    total=0j
    # Sum only old product-m states with nonzero intertwiner amplitude.
    for ms in product(*[mvals(s) for s in old_spins]):
        io=intertwiner_amplitude(old_spins,tuple(old_path),ms)
        if io==0: continue
        # untouched new magnetic indices equal old ones
        for a2 in mvals(r2):
            b2=-a2; ph=eps_phase(r2,a2)
            for Mi in mvals(new_spins[leaf_i]):
                c1=cg(old_spins[leaf_i],ms[leaf_i],r2,a2,new_spins[leaf_i],Mi)
                if c1==0: continue
                for Mj in mvals(new_spins[leaf_j]):
                    c2=cg(old_spins[leaf_j],ms[leaf_j],r2,b2,new_spins[leaf_j],Mj)
                    if c2==0: continue
                    Ms=list(ms);Ms[leaf_i]=Mi;Ms[leaf_j]=Mj;Ms=tuple(Ms)
                    inn=intertwiner_amplitude(new_spins,tuple(new_path),Ms)
                    if inn!=0:
                        total += np.conjugate(inn)*io*ph*c1*c2
    return complex(total)


def link_factor(old_j2,new_j2):
    return math.sqrt((old_j2+1)/(new_j2+1))


def vacuum_corner_check(r2:int, pair=(0,1)):
    old=(0,0,0,0,0,0); op=fusion_paths(old)[0]
    new=list(old);new[pair[0]]=r2;new[pair[1]]=r2;new=tuple(new)
    ps=fusion_paths(new); assert len(ps)==1
    z=corner_element(old,op,new,ps[0],pair[0],pair[1],r2)
    return z, math.sqrt(r2+1)



def corner_element_dp(old_spins,old_path,new_spins,new_path,leaf_i,leaf_j,r2):
    """Dynamic-programming evaluation of the same corner tensor.

    Complexity is polynomial in magnetic dimensions and never materializes the
    six-leg product basis.  This is the production primitive for j up to the
    certified fine cutoff.
    """
    old_spins=tuple(old_spins); new_spins=tuple(new_spins)
    old_path=tuple(old_path); new_path=tuple(new_path)
    if leaf_i==leaf_j: raise ValueError('distinct leaves required')
    i,j=sorted((leaf_i,leaf_j))
    if any(old_spins[k]!=new_spins[k] for k in range(6) if k not in (i,j)): return 0j
    if new_spins[i] not in fuse(old_spins[i],r2) or new_spins[j] not in fuse(old_spins[j],r2): return 0j

    def leaf_transitions(k,pending):
        jo,jn=old_spins[k],new_spins[k]
        if k not in (i,j):
            for m in mvals(jo):
                if m in mvals(jn):
                    yield m,m,pending,1.0
        elif k==i:
            if pending is not None: raise RuntimeError('unexpected pending line')
            for a in mvals(r2):
                for mo in mvals(jo):
                    for mn in mvals(jn):
                        c=cg(jo,mo,r2,a,jn,mn)
                        if c: yield mo,mn,a,c
        else:
            if pending is None: return
            a=pending; b=-a; ph=eps_phase(r2,a)
            for mo in mvals(jo):
                for mn in mvals(jn):
                    c=cg(jo,mo,r2,b,jn,mn)
                    if c: yield mo,mn,None,ph*c

    # k=0 initializes the two chain magnetic numbers.
    states={}
    for mo,mn,pend,w in leaf_transitions(0,None):
        states[(mo,mn,pend)]=states.get((mo,mn,pend),0j)+w
    for k in range(1,6):
        Aold_prev=old_spins[0] if k==1 else old_path[k-2]
        Anew_prev=new_spins[0] if k==1 else new_path[k-2]
        Aold=old_path[k-1]; Anew=new_path[k-1]
        nxt={}
        for (Mop,Mnp,pend),amp in states.items():
            for mo,mn,pend2,w in leaf_transitions(k,pend):
                Mo=Mop+mo; Mn=Mnp+mn
                co=cg(Aold_prev,Mop,old_spins[k],mo,Aold,Mo)
                if co==0: continue
                cn=cg(Anew_prev,Mnp,new_spins[k],mn,Anew,Mn)
                if cn==0: continue
                key=(Mo,Mn,pend2)
                nxt[key]=nxt.get(key,0j)+amp*w*co*np.conjugate(cn)
        states=nxt
        if not states: return 0j
    return complex(states.get((0,0,None),0j))

def projected_jhalf_transition_table():
    """All nonzero r=1 local transitions with external spins in {0,1}."""
    rows=[]
    for old in product((0,1),repeat=6):
      ops=fusion_paths(old)
      if not ops: continue
      for i in range(6):
       for j in range(i+1,6):
        for new_i in fuse(old[i],1):
         if new_i not in (0,1): continue
         for new_j in fuse(old[j],1):
          if new_j not in (0,1): continue
          new=list(old);new[i]=new_i;new[j]=new_j;new=tuple(new)
          nps=fusion_paths(new)
          if not nps: continue
          for oi,op in enumerate(ops):
           for ni,npth in enumerate(nps):
            z=corner_element(old,op,new,npth,i,j,1)
            if abs(z)>1e-13:
             rows.append((old,oi,i,j,new,ni,float(z.real),float(z.imag)))
    return rows


def self_test():
    vac=[]
    for r2 in (1,2,3):
        z,target=vacuum_corner_check(r2)
        err=abs(z-target); assert err<2e-14
        # Four vacuum corners and four normalized links -> unit Wilson-loop coefficient.
        loop=(z**4)*(link_factor(0,r2)**4)
        lerr=abs(loop-1); assert lerr<5e-14
        vac.append({'twice_r':r2,'corner':[z.real,z.imag],'target_sqrt_dim':target,
                    'single_loop_vacuum_amplitude':[loop.real,loop.imag],'loop_residual':lerr})
    # Dense-vs-DP exact regression on all j<=1/2 local transitions.
    dpmax=0.0; dptests=0
    for old in product((0,1),repeat=6):
      for op in fusion_paths(old):
       for i in range(6):
        for j in range(i+1,6):
         for ni in fuse(old[i],1):
          if ni not in (0,1): continue
          for nj in fuse(old[j],1):
           if nj not in (0,1): continue
           new=list(old);new[i]=ni;new[j]=nj;new=tuple(new)
           for npth in fusion_paths(new):
            z0=corner_element(old,op,new,npth,i,j,1)
            z1=corner_element_dp(old,op,new,npth,i,j,1)
            dpmax=max(dpmax,abs(z0-z1));dptests+=1
    assert dpmax<3e-14
    # High-spin production sanity: no dense six-leg tensor is constructed.
    hs=(4,3,2,3,4,2); hp=fusion_paths(hs)
    high_spin_sample=None
    if hp:
      # choose a transition guaranteed by fusion on leaves 1 and 4 when available
      for n1 in fuse(hs[1],3):
       for n4 in fuse(hs[4],3):
        ns=list(hs);ns[1]=n1;ns[4]=n4;ns=tuple(ns); nps=fusion_paths(ns)
        if nps:
         high_spin_sample=corner_element_dp(hs,hp[0],ns,nps[0],1,4,3); break
       if high_spin_sample is not None: break
    rows=projected_jhalf_transition_table()
    # Reciprocity of the local invariant insertion after including the two local
    # endpoint link-normalization square roots.  This is a diagnostic rather
    # than the full plaquette Hermiticity statement.
    lookup={}
    for old,oi,i,j,new,ni,re,im in rows:
        lookup[(old,oi,i,j,new,ni)]=complex(re,im)
    maxrec=0.0; tested=0
    for key,z in lookup.items():
        old,oi,i,j,new,ni=key
        rev=lookup.get((new,ni,i,j,old,oi))
        if rev is not None:
            f=math.sqrt(link_factor(old[i],new[i])*link_factor(old[j],new[j]))
            fr=math.sqrt(link_factor(new[i],old[i])*link_factor(new[j],old[j]))
            maxrec=max(maxrec,abs(abs(f*z)-abs(fr*rev)));tested+=1
    # Fusion-tree basis phases can change signs under reversal; Hermiticity is
    # recovered after the corresponding basis-gauge phase is carried globally.
    # The invariant local magnitude must be reciprocal independent of that gauge.
    assert maxrec<2e-13
    return {
      'status':'PASS_SU2_FUSION_TREE_PLAQUETTE_CORNER',
      'vacuum_character_loop_checks':vac,
      'dense_vs_dp_tests':dptests,
      'dense_vs_dp_max_residual':dpmax,
      'high_spin_dp_sample':None if high_spin_sample is None else [high_spin_sample.real,high_spin_sample.imag],
      'jhalf_projected_nonzero_local_transitions':len(rows),
      'reciprocity_pairs_tested':tested,
      'max_local_reciprocity_magnitude_residual':maxrec,
      'normalization':'Vertex coefficient excludes one sqrt(d_old/d_new) Peter-Weyl factor per affected physical link; the product of four corner coefficients and four link factors gives unit amplitude for chi_r(U_p)|vac>.',
      'global_use':'Cache these corner blocks (or their equivalent 6j form) and assemble each plaquette update without magnetic-index storage.',
      'scope':'Exact local plaquette-corner primitive. Full 24-link repeated-time PEPO sweep remains a separate contraction.'
    }

if __name__=='__main__': print(json.dumps(self_test(),indent=2))
