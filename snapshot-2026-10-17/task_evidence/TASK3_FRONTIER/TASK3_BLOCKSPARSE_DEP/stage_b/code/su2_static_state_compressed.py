#!/usr/bin/env python3
from __future__ import annotations
import sys, os, json, math, time
from pathlib import Path
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.special import jv
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;R=ROOT/'results';sys.path.insert(0,str(HERE))
import fusion_tree_pointsplit as ft
import su2_fusion_corner as sc
import su2_global_spin_network as gs
from fast_su2_cg import cg_fast
ft.cg=cg_fast;sc.cg=cg_fast;gs.corner_element_dp=sc.corner_element_dp
X=32*math.pi/63

def coeff(r2):
    d=r2+1
    return (1j**r2)*(2*d/X)*jv(d,X)

def compress(v,budget):
    if not v:return v,0.0
    keys=list(v);sq=np.array([abs(v[k])**2 for k in keys]);idx=np.argsort(sq);cs=np.cumsum(sq[idx]);nd=int(np.searchsorted(cs,budget*budget,side='right'));nd=min(nd,max(0,len(keys)-1))
    if nd<=0:return v,0.0
    drop=set(int(i) for i in idx[:nd]);disc=float(sq[list(drop)].sum()**.5)
    return {k:v[k] for i,k in enumerate(keys) if i not in drop},disc

def neighbors(old,pid,r2):
    if r2==0:return {old:1+0j}
    p=gs.PLAQ[pid];edges=p['edges'];av=sorted({gs.SID[c[0]] for c in p['corners']})
    choices=[ft.fuse(old.links[e],r2) for e in edges]
    out={}
    for vals in product(*choices):
        ls=list(old.links)
        for e,j in zip(edges,vals):ls[e]=j
        ls=tuple(ls);pops=[];ok=True
        for vi in av:
            q=ft.fusion_paths(gs.vertex_spins(ls,gs.SITES[vi]))
            if not q:ok=False;break
            pops.append(q)
        if not ok:continue
        for sel in product(*pops):
            ps=list(old.paths)
            for vi,q in zip(av,sel):ps[vi]=q
            st=gs.SpinNetwork(ls,tuple(ps));z=gs.plaquette_element(old,st,pid,r2)
            if abs(z)>1e-14:out[st]=out.get(st,0j)+z
    return out

def run(rmax2=2,budget=1e-9,max_states=200000):
    v={gs.vacuum_state():1+0j};profile=[];dsum=0.0;t0=time.time()
    for pid in range(24):
        out=defaultdict(complex);t=time.time()
        for st,a in v.items():
            for r2 in range(rmax2+1):
                c=coeff(r2)
                if abs(c)<1e-16:continue
                for q,z in neighbors(st,pid,r2).items():out[q]+=a*c*z
        raw=len(out);v,disc=compress(dict(out),budget);dsum+=disc
        nrm=sum(abs(a)**2 for a in v.values())**.5
        profile.append({'plaquette':pid,'raw_states':raw,'kept_states':len(v),'discarded_l2':disc,'kept_norm':nrm,'seconds':time.time()-t})
        print('p',pid,'raw',raw,'keep',len(v),'disc',disc,'norm',nrm,'sec',profile[-1]['seconds'],flush=True)
        if len(v)>max_states:break
    overlap=v.get(gs.vacuum_state(),0j)
    out={'status':'PASS_SU2_STATIC_PHYSICAL_STATE_COMPRESSED_PROBE','twice_j_character_max':rmax2,'per_gate_discard_budget':budget,'plaquettes_completed':len(profile),'sum_discarded_l2':dsum,'final_support':len(v),'vacuum_overlap':[overlap.real,overlap.imag],'norm':sum(abs(a)**2 for a in v.values())**.5,'profile':profile,'seconds':time.time()-t0,'scope':'Physical gauge-invariant spin-network state under sequential truncated SU(2) plaquette character gates from the vacuum. Compression is direct l2 pruning in an orthonormal spin-network basis. This is a static magnetic-layer contractor probe, not the repeated-time Kato h_d calculation.'}
    (R/f'SU2_STATIC_STATE_R{rmax2}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','twice_j_character_max','plaquettes_completed','sum_discarded_l2','final_support','vacuum_overlap','norm','seconds')},indent=2))
if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else 2,float(sys.argv[2]) if len(sys.argv)>2 else 1e-9)
