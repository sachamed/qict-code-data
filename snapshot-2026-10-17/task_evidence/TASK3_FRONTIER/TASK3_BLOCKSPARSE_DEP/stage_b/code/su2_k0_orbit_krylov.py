#!/usr/bin/env python3
from __future__ import annotations
import sys, os, json, time, math
from functools import lru_cache
from itertools import product
from collections import defaultdict
sys.path.insert(0,os.path.dirname(__file__))
import fusion_tree_pointsplit as ft
import su2_fusion_corner as sc
import su2_global_spin_network as gs
from fast_su2_cg import cg_fast
ft.cg=cg_fast; sc.cg=cg_fast
SpinNetwork=gs.SpinNetwork
L=2

@lru_cache(maxsize=None)
def fpaths(spins): return tuple(ft.fusion_paths(tuple(spins)))
_corner_raw=sc.corner_element_dp
@lru_cache(maxsize=None)
def corner_cached(os_,op,ns,npth,i,j,r2=1):
    return _corner_raw(os_,op,ns,npth,i,j,r2)
sc.corner_element_dp=corner_cached; gs.corner_element_dp=corner_cached

TRANSLATIONS=[(dx,dy,dz) for dx in range(2) for dy in range(2) for dz in range(2)]
def addsite(s,d): return tuple((s[i]+d[i])%2 for i in range(3))
@lru_cache(maxsize=None)
def translate(st:SpinNetwork,d):
    nl=[0]*24
    for eid,(mu,x,y,z) in enumerate(gs.LINKS):
        s=(x,y,z); t=addsite(s,d); ne=gs.LID[(mu,*t)]; nl[ne]=st.links[eid]
    np=[None]*8
    for v in gs.SITES:
        nv=addsite(v,d); np[gs.SID[nv]]=st.paths[gs.SID[v]]
    return SpinNetwork(tuple(nl),tuple(np))

def skey(st): return (st.links,st.paths)
@lru_cache(maxsize=None)
def orbit(st):
    dd={skey(translate(st,d)):translate(st,d) for d in TRANSLATIONS}
    return tuple(dd[k] for k in sorted(dd))
@lru_cache(maxsize=None)
def canonical(st): return orbit(st)[0]

@lru_cache(maxsize=None)
def wp_action(st:SpinNetwork,pid:int):
    p=gs.PLAQ[pid]; es=p['edges']; av=sorted({gs.SID[c[0]] for c in p['corners']})
    choices=[]
    for e in es:
        choices.append(tuple(ft.fuse(st.links[e],1)))
    out=defaultdict(complex)
    for vals in product(*choices):
        nl=list(st.links)
        for e,v in zip(es,vals): nl[e]=v
        nl=tuple(nl)
        vp=[]; ok=True
        for vi in av:
            ps=fpaths(gs.vertex_spins(nl,gs.SITES[vi]))
            if not ps: ok=False; break
            vp.append(ps)
        if not ok: continue
        for pp in product(*vp):
            paths=list(st.paths)
            for vi,pth in zip(av,pp): paths[vi]=pth
            ns=SpinNetwork(nl,tuple(paths))
            z=gs.plaquette_element(st,ns,pid,1)
            if abs(z)>1e-14: out[ns]+=z
    return tuple(out.items())

def H_on_orbit_vector(vec):
    # vec stores the common coefficient of each individual state in a translation orbit.
    # Translation covariance permits an exact representative-only update:
    # b_common(O_b)=sum_a a_common(O_a)*(|O_a|/|O_b|)*sum_{t in O_b}<t|H|a_rep>.
    agg=defaultdict(complex)
    for rep,a in vec.items():
        na=len(orbit(rep))
        local=defaultdict(complex)
        for pid in range(24):
            for tgt,z in wp_action(rep,pid):
                local[canonical(tgt)] += z
        for crep,z in local.items():
            nb=len(orbit(crep))
            agg[crep] += a*(na/nb)*z
    return {rep:z for rep,z in agg.items() if abs(z)>1e-14}

def norm2(vec): return float(sum(len(orbit(s))*abs(a)**2 for s,a in vec.items()))
def inner(a,b): return sum(len(orbit(s))*complex(a.get(s,0)).conjugate()*v for s,v in b.items())

def self_test(depth=3):
    vac=canonical(gs.vacuum_state()); vec={vac:1+0j}; moments=[1+0j]; supports=[1]; norms=[1.0]; timings=[]
    for d in range(1,depth+1):
        t=time.perf_counter(); vec=H_on_orbit_vector(vec); timings.append(time.perf_counter()-t)
        supports.append(len(vec)); norms.append(norm2(vec)); moments.append(vec.get(vac,0j))
    # H|0> consists of 24 unit-amplitude one-plaquette loops -> <0|H^2|0>=24.
    m2=moments[2] if depth>=2 else None
    if depth>=2 and abs(m2-24)>2e-10: raise AssertionError(('m2',m2))
    # odd first moment vanishes.
    if abs(moments[1])>1e-13: raise AssertionError(moments[1])
    return {
      'status':'PASS_K0_TRANSLATION_ORBIT_HB_KRYLOV', 'depth':depth,
      'orbit_support_by_depth':supports,
      'vector_norm_sq_by_depth':norms,
      'vacuum_moments_HB_power':[[z.real,z.imag] for z in moments],
      'm2_residual_from_24':None if m2 is None else abs(m2-24),
      'seconds_by_matvec':timings,
      'max_cached_CG':cg_fast.cache_info().currsize,
      'cached_plaquette_actions':wp_action.cache_info().currsize,
      'representation':'translation-orbit coefficients in exact point-split spin-network basis; no magnetic indices',
      'scope':'Exact Krylov action of H_B=sum_p chi_1/2(U_p). Electric kernel, C8 and C10 are not included in this self-test.'
    }
if __name__=='__main__':
    depth=int(sys.argv[1]) if len(sys.argv)>1 else 3
    print(json.dumps(self_test(depth),indent=2))
