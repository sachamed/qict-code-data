#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json, math, sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from fusion_tree_pointsplit import fusion_paths, fuse, mvals, cg
from su2_global_spin_network import SITES,SID,LINKS,LID,PLAQ,shift,leaf_for_link_at_vertex,link_factor
from su2_fusion_corner import eps_phase

ROOT=HERE.parent
OUT=ROOT/'results'/'SU2_CHARGED_OPENLINE_SEED.json'
SRC=(0,0,0); SNK=(1,1,1); MATTER2=1

@staticmethod
def _dummy(): pass

@dataclass(frozen=True)
class ChargedSN:
    links: tuple[int,...]
    paths: tuple[tuple[int,...],...]

def local_spins(link_spins,v):
    vals=[]
    for mu in range(3): vals.append(link_spins[LID[(mu,*v)]])
    for mu in range(3):
        sm=shift(v,mu,-1); vals.append(link_spins[LID[(mu,*sm)]])
    if v in (SRC,SNK): vals.append(MATTER2)
    return tuple(vals)

def generic_corner_dp(old_spins,old_path,new_spins,new_path,leaf_i,leaf_j,r2):
    old_spins=tuple(old_spins); new_spins=tuple(new_spins); old_path=tuple(old_path); new_path=tuple(new_path)
    N=len(old_spins)
    if len(new_spins)!=N or N<2 or leaf_i==leaf_j: raise ValueError('bad local data')
    i,j=sorted((leaf_i,leaf_j))
    if any(old_spins[k]!=new_spins[k] for k in range(N) if k not in (i,j)): return 0j
    if new_spins[i] not in fuse(old_spins[i],r2) or new_spins[j] not in fuse(old_spins[j],r2): return 0j
    def trans(k,pending):
        jo,jn=old_spins[k],new_spins[k]
        if k not in (i,j):
            for m in mvals(jo):
                if m in mvals(jn): yield m,m,pending,1.0
        elif k==i:
            if pending is not None: raise RuntimeError
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
    states={}
    for mo,mn,p,w in trans(0,None): states[(mo,mn,p)]=states.get((mo,mn,p),0j)+w
    for k in range(1,N):
        Ao_prev=old_spins[0] if k==1 else old_path[k-2]
        An_prev=new_spins[0] if k==1 else new_path[k-2]
        Ao=old_path[k-1]; An=new_path[k-1]
        nxt={}
        for (Mop,Mnp,p),amp in states.items():
            for mo,mn,p2,w in trans(k,p):
                Mo=Mop+mo; Mn=Mnp+mn
                co=cg(Ao_prev,Mop,old_spins[k],mo,Ao,Mo)
                if co==0: continue
                cn=cg(An_prev,Mnp,new_spins[k],mn,An,Mn)
                if cn==0: continue
                key=(Mo,Mn,p2); nxt[key]=nxt.get(key,0j)+amp*w*co*np.conjugate(cn)
        states=nxt
        if not states:return 0j
    return complex(states.get((0,0,None),0j))

def canonical_paths(ls):
    ps=[]
    for v in SITES:
        q=fusion_paths(local_spins(ls,v))
        if len(q)!=1: return None
        ps.append(q[0])
    return tuple(ps)

def seed(order=(2,1,0)):
    ls=[0]*24; s=SRC
    for mu in order:
        eid=LID[(mu,*s)]; ls[eid]=1; s=shift(s,mu)
    if s!=SNK: raise AssertionError
    ps=canonical_paths(tuple(ls))
    if ps is None: raise RuntimeError('reference open line not unique')
    return ChargedSN(tuple(ls),ps)

def plaquette_element(old,new,pid,r2=1):
    p=PLAQ[pid]; affected=set(p['edges'])
    if any(old.links[e]!=new.links[e] for e in range(24) if e not in affected): return 0j
    for e in affected:
        if new.links[e] not in fuse(old.links[e],r2): return 0j
    av={SID[c[0]] for c in p['corners']}
    if any(old.paths[v]!=new.paths[v] for v in range(8) if v not in av): return 0j
    z=1+0j
    for e in p['edges']: z*=link_factor(old.links[e],new.links[e])
    for v,e1,e2 in p['corners']:
        vi=SID[v]; os=local_spins(old.links,v); ns=local_spins(new.links,v)
        i=leaf_for_link_at_vertex(e1,v); j=leaf_for_link_at_vertex(e2,v)
        z*=generic_corner_dp(os,old.paths[vi],ns,new.paths[vi],i,j,r2)
    return complex(z)

def neighbors(old,pid,r2=1,j2max=2):
    p=PLAQ[pid]; edges=p['edges']; av=sorted({SID[c[0]] for c in p['corners']})
    choices=[]
    for e in edges: choices.append([j for j in fuse(old.links[e],r2) if j<=j2max])
    out={}
    for vals in product(*choices):
        ls=list(old.links)
        for e,j in zip(edges,vals): ls[e]=j
        ls=tuple(ls)
        path_opts=[]; ok=True
        for vi in av:
            qs=fusion_paths(local_spins(ls,SITES[vi]))
            if not qs: ok=False; break
            path_opts.append(qs)
        if not ok: continue
        for sel in product(*path_opts):
            ps=list(old.paths)
            for vi,q in zip(av,sel): ps[vi]=q
            st=ChargedSN(ls,tuple(ps)); z=plaquette_element(old,st,pid,r2)
            if abs(z)>1e-14: out[st]=out.get(st,0j)+z
    return out

def main():
    s0=seed((2,1,0)); salt=seed((0,1,2))
    endpoint_dims={str(v):len(fusion_paths(local_spins(s0.links,v))) for v in (SRC,SNK)}
    rows=[]; maxherm=0.; H={}
    for pid in range(24):
        nb=neighbors(s0,pid,1,2)
        ph=0.0
        for st,z in nb.items():
            zr=plaquette_element(st,s0,pid,1)
            ph=max(ph,abs(z-np.conjugate(zr)))
            H[st]=H.get(st,0j)+z
        maxherm=max(maxherm,ph)
        rows.append({'plaquette':pid,'nonzero_neighbors':len(nb),'hermiticity_max_residual':float(ph),'out_norm2':float(sum(abs(z)**2 for z in nb.values()))})
    hdiag=H.get(s0,0j)
    hb_norm2=float(sum(abs(z)**2 for z in H.values()))
    # Check endpoint Gauss target for every generated state by existence of a singlet fusion path including matter leg.
    bad=0
    for st in H:
        for v in SITES:
            if st.paths[SID[v]] not in fusion_paths(local_spins(st.links,v)): bad+=1
    checks={
      'canonical_open_line_reaches_111':True,
      'charged_endpoint_intertwiner_unique':bool(endpoint_dims[str(SRC)]==1 and endpoint_dims[str(SNK)]==1),
      'all_generated_states_satisfy_exact_charged_gauss':bool(bad==0),
      'plaquette_block_hermiticity':bool(maxherm<2e-12),
      'nonzero_HB_seed_image':bool(hb_norm2>0),
    }
    out={
      'status':'PASS_SU2_CHARGED_OPENLINE_FUSION_SEED' if all(checks.values()) else 'FAIL',
      'source':list(SRC),'sink':list(SNK),'matter_twice_j_at_endpoints':1,
      'canonical_splitstep_path_order':['z','y','x'],
      'canonical_path_link_ids':[i for i,j in enumerate(s0.links) if j==1],
      'alternative_xyz_path_link_ids':[i for i,j in enumerate(salt.links) if j==1],
      'representation':'Six gauge leaves at every cubic vertex; at source and sink a seventh fundamental matter leaf is appended. Every local tree ends in total singlet, so charged Gauss law is exact without magnetic-index storage.',
      'endpoint_local_fusion_path_counts':endpoint_dims,
      'unique_states_in_HB_seed_image':len(H),
      'HB_seed_diagonal':[float(hdiag.real),float(hdiag.imag)],
      'HB_seed_norm2':hb_norm2,
      'max_forward_reverse_plaquette_residual':float(maxherm),
      'rows':rows,'checks':checks,
      'next_step':'Use this charged spin-network seed and plaquette matvec in a Riesz/Krylov continuation of the SU(2) core branch; couple it conditionally to the already-closed U(1) affine Kato sector through the shared C8 branch index.'
    }
    OUT.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
