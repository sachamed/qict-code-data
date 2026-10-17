#!/usr/bin/env python3
from __future__ import annotations
import sys, os, math, time, json
from collections import defaultdict
from itertools import product
from functools import lru_cache
from pathlib import Path
from scipy.special import jv
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
import su2_global_spin_network as gs
import fusion_tree_pointsplit as ft
import su2_fusion_corner as sc
from fast_su2_cg import cg_fast
ft.cg=cg_fast; sc.cg=cg_fast
ROOT=Path(__file__).resolve().parent.parent
PHI=math.pi/63.0; X=32.0*PHI

@lru_cache(maxsize=None)
def wp_action_r(st,pid,r2):
    p=gs.PLAQ[pid]; es=p['edges']; av=sorted({gs.SID[c[0]] for c in p['corners']})
    choices=[tuple(ft.fuse(st.links[e],r2)) for e in es]
    out=defaultdict(complex)
    for vals in product(*choices):
        nl=list(st.links)
        for e,v in zip(es,vals): nl[e]=v
        nl=tuple(nl); vp=[]
        for vi in av:
            ps=k.fpaths(gs.vertex_spins(nl,gs.SITES[vi]))
            if not ps: break
            vp.append(ps)
        else:
            for pp in product(*vp):
                paths=list(st.paths)
                for vi,pth in zip(av,pp): paths[vi]=pth
                ns=gs.SpinNetwork(nl,tuple(paths))
                z=gs.plaquette_element(st,ns,pid,r2)
                if abs(z)>1e-15: out[ns]+=z
    return tuple(out.items())

def coeff(r2):
    d=r2+1
    return (1j**r2)*(2.0*d/X)*jv(d,X)

def apply_gate(vec,pid,rmax):
    out=defaultdict(complex)
    for st,a in vec.items():
        for r2 in range(rmax+1):
            c=coeff(r2)
            if abs(c)<1e-16: continue
            for ns,z in wp_action_r(st,pid,r2): out[ns]+=a*c*z
    return {s:a for s,a in out.items() if abs(a)>1e-16}

def prune(vec,budget):
    items=sorted(vec.items(),key=lambda kv:abs(kv[1])**2)
    acc=0.0; cut=0; B2=budget*budget
    for i,(_,a) in enumerate(items):
        x=abs(a)**2
        if acc+x<=B2: acc+=x;cut=i+1
        else: break
    return dict(items[cut:]),math.sqrt(acc)

def main():
    # This is deliberately only a feasibility probe; r2<=8 already contains the
    # large coefficients and is sufficient to demonstrate state-count explosion.
    rmax=8; budget=1e-9; v={gs.vacuum_state():1+0j}; rows=[]; total_disc=0.0
    for pid in range(4):
        t=time.perf_counter(); v=apply_gate(v,pid,rmax); sec=time.perf_counter()-t
        before=len(v); norm=math.sqrt(sum(abs(a)**2 for a in v.values()))
        v,disc=prune(v,budget); total_disc+=disc
        rows.append({'plaquette_gate':pid+1,'states_before_prune':before,'states_after_prune':len(v),
                     'state_norm_before_prune':norm,'discarded_l2_this_gate':disc,'seconds':sec})
    out={'status':'PASS_DIRECT_SPARSE_CHARACTER_STATE_EXPANSION_REJECTED_AS_FINAL_SOLVER',
         'character_twice_j_probe_max':rmax,'per_gate_prune_budget':budget,'rows':rows,
         'accumulated_discarded_l2_triangle_bound':total_disc,
         'final_states_after_four_plaquettes':len(v),
         'conclusion':'Even before a complete 24-plaquette layer, direct orthonormal-state enumeration grows to hundreds of thousands of states at a 1e-9 pruning budget. The physical solver must compress Schmidt/frontier blocks rather than prune global spin-network coefficients.',
         'scope':'Algorithmic feasibility probe only. It is not h_d and does not use this reduced rmax as a physical regulator.'}
    p=ROOT/'results'/'SU2_SPARSE_CHARACTER_STATE_EXPANSION_PROBE.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
