#!/usr/bin/env python3
"""Global L=2 SU(2) spin-network plaquette primitive for Task-3 Stage B.

This file lifts the exact fusion-tree corner tensor to the full 24-link periodic
cubic graph.  It provides matrix elements of one character Wilson plaquette
between gauge-invariant spin-network basis states without magnetic indices.
The global PEPO solver can use this as its sparse local update.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import json, math
import numpy as np
from fusion_tree_pointsplit import fusion_paths, fuse
from su2_fusion_corner import corner_element_dp, link_factor

L=2
SITES=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
SID={s:i for i,s in enumerate(SITES)}
LINKS=[(mu,x,y,z) for mu in range(3) for x,y,z in SITES]
LID={e:i for i,e in enumerate(LINKS)}

def shift(s,mu,sgn=1):
    a=list(s);a[mu]=(a[mu]+sgn)%L;return tuple(a)

def link_endpoints(eid):
    mu,x,y,z=LINKS[eid];s=(x,y,z);return s,shift(s,mu,+1)

def leaf_for_link_at_vertex(eid,v):
    mu,x,y,z=LINKS[eid];s=(x,y,z);t=shift(s,mu,+1)
    if v==s:return mu          # +x,+y,+z
    if v==t:return 3+mu        # -x,-y,-z
    raise ValueError('link not incident on vertex')

def plaquettes():
    out=[]
    for mu in range(3):
      for nu in range(mu+1,3):
       for s in SITES:
        sm=shift(s,mu);sn=shift(s,nu);smn=shift(sm,nu)
        es=[LID[(mu,*s)],LID[(nu,*sm)],LID[(mu,*sn)],LID[(nu,*s)]]
        signs=[+1,+1,-1,-1]
        verts=[s,sm,smn,sn]
        # corners: v0 joins e3,e0; v1 e0,e1; v2 e1,e2; v3 e2,e3
        corners=[(verts[0],es[3],es[0]),(verts[1],es[0],es[1]),
                 (verts[2],es[1],es[2]),(verts[3],es[2],es[3])]
        out.append({'mu':mu,'nu':nu,'site':s,'edges':es,'signs':signs,'corners':corners})
    return out
PLAQ=plaquettes()

@dataclass(frozen=True)
class SpinNetwork:
    links: tuple[int,...]                 # 24 twice-spins
    paths: tuple[tuple[int,...],...]      # 8 fusion paths

def vertex_spins(link_spins,v):
    vals=[]
    for mu in range(3): vals.append(link_spins[LID[(mu,*v)]])
    for mu in range(3):
        sm=shift(v,mu,-1); vals.append(link_spins[LID[(mu,*sm)]])
    return tuple(vals)

def vacuum_state():
    ls=(0,)*24
    ps=tuple(fusion_paths(vertex_spins(ls,v))[0] for v in SITES)
    return SpinNetwork(ls,ps)

def canonical_paths_for_links(ls):
    arr=[]
    for v in SITES:
        p=fusion_paths(vertex_spins(ls,v))
        if len(p)!=1: return None
        arr.append(p[0])
    return tuple(arr)

def single_plaquette_loop(pid,r2=1):
    ls=[0]*24
    for e in PLAQ[pid]['edges']: ls[e]=r2
    ps=canonical_paths_for_links(tuple(ls))
    if ps is None: raise RuntimeError('single loop should have unique intertwiners')
    return SpinNetwork(tuple(ls),ps)

def plaquette_element(old:SpinNetwork,new:SpinNetwork,pid:int,r2:int)->complex:
    p=PLAQ[pid]; affected=set(p['edges'])
    if any(old.links[e]!=new.links[e] for e in range(24) if e not in affected): return 0j
    for e in affected:
        if new.links[e] not in fuse(old.links[e],r2): return 0j
    # Unaffected vertices must retain the same fusion path.  A vertex is affected
    # exactly when it is a corner of this plaquette.
    av={SID[c[0]] for c in p['corners']}
    if any(old.paths[v]!=new.paths[v] for v in range(8) if v not in av): return 0j
    z=1+0j
    for e in p['edges']: z*=link_factor(old.links[e],new.links[e])
    for v,e1,e2 in p['corners']:
        vi=SID[v]; os=vertex_spins(old.links,v); ns=vertex_spins(new.links,v)
        i=leaf_for_link_at_vertex(e1,v);j=leaf_for_link_at_vertex(e2,v)
        z*=corner_element_dp(os,old.paths[vi],ns,new.paths[vi],i,j,r2)
    return complex(z)

def self_test():
    vac=vacuum_state(); loops=[single_plaquette_loop(p,1) for p in range(24)]
    fwd=[];rev=[]
    for p,s in enumerate(loops):
        a=plaquette_element(vac,s,p,1);b=plaquette_element(s,vac,p,1)
        fwd.append(abs(a-1));rev.append(abs(b-1))
    assert max(fwd)<2e-14 and max(rev)<2e-14
    # Projected Hermiticity on span{vac, 24 single-loop states}.  Matrix elements
    # to states outside the span are irrelevant to this test; the retained block
    # itself must be Hermitian.
    states=[vac]+loops; maxh=0.0; nonzero=0
    for p in range(24):
      M=np.zeros((len(states),len(states)),complex)
      for i,a in enumerate(states):
       for j,b in enumerate(states):
        z=plaquette_element(b,a,p,1) # <a|W_p|b>
        M[i,j]=z
        if abs(z)>1e-14:nonzero+=1
      maxh=max(maxh,float(np.linalg.norm(M-M.conj().T,2)))
    assert maxh<5e-13
    # Geometry checks.
    incid=[0]*24
    for p in PLAQ:
      for e in p['edges']:incid[e]+=1
    assert len(PLAQ)==24 and set(incid)=={4}
    return {
      'status':'PASS_GLOBAL_L2_SU2_SPIN_NETWORK_PLAQUETTE_PRIMITIVE',
      'geometry':{'sites':8,'links':24,'plaquettes':24,'link_plaquette_valence':4},
      'max_vacuum_to_single_loop_residual':max(fwd),
      'max_single_loop_to_vacuum_residual':max(rev),
      'projected_25state_plaquette_hermiticity_2norm_residual':maxh,
      'nonzero_projected_matrix_elements_tested':nonzero,
      'representation':'24 physical link twice-spins + one fixed fusion-tree path at each of 8 vertices; no magnetic indices.',
      'scope':'Exact global one-plaquette sparse matrix-element engine. Repeated product of 24 character gates and time-electric evolution is the next layer.'
    }

if __name__=='__main__':print(json.dumps(self_test(),indent=2))
