#!/usr/bin/env python3
from __future__ import annotations
import sys, os, math, cmath, json, time
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np
from scipy.special import jv

UP='/mnt/data/task3_resume'
sys.path.insert(0,UP)
import fusion_tree_pointsplit as ft
import su2_fusion_corner as sc
import su2_global_spin_network as gs
from fast_su2_cg import cg_fast
ft.cg=cg_fast; sc.cg=cg_fast
cg=cg_fast; fuse=ft.fuse; mvals=ft.mvals
PHI=math.pi/63.0; X=32.0*PHI
# Certified optimal link order from archive.
ORDER=[12,13,9,8,3,1,0,2,22,20,16,18,4,5,7,6,21,23,17,19,15,11,14,10]
POS={e:i for i,e in enumerate(ORDER)}

# For each vertex choose the exact point-split fusion tree adapted to the global link sweep.
INC_ORDER={}
LOCAL_RANK={}
for vi,v in enumerate(gs.SITES):
    inc=[]
    for mu in range(3):
        inc.append(gs.LID[(mu,*v)])
        inc.append(gs.LID[(mu,*gs.shift(v,mu,-1))])
    inc=tuple(sorted(set(inc), key=lambda e:POS[e]))
    assert len(inc)==6
    INC_ORDER[vi]=inc
    for k,e in enumerate(inc): LOCAL_RANK[(vi,e)]=k

# Endpoint vertices of each link.
ENDS={e:tuple(gs.SID[x] for x in gs.link_endpoints(e)) for e in range(24)}

# Site physical label = (j2, intermediate channel at endpoint0 if created, endpoint1 if created)
# Endpoint order is increasing vertex id for deterministic serialization.
def phys_vertices(e): return tuple(sorted(ENDS[e]))

def created_channel(vi,e):
    k=LOCAL_RANK[(vi,e)]
    return 1 <= k <= 4

def pack_phys(e,j,new_after_by_v):
    vals=[int(j)]
    for vi in phys_vertices(e):
        if created_channel(vi,e): vals.append(int(new_after_by_v[vi]))
    return tuple(vals)

def unpack_phys(e,phys,left_sector):
    j=int(phys[0]); it=iter(phys[1:]); right=list(left_sector)
    for vi in phys_vertices(e):
        k=LOCAL_RANK[(vi,e)]
        if k==0:
            right[vi]=j
        elif 1<=k<=4:
            right[vi]=int(next(it))
        else:
            # last incident link closes singlet
            right[vi]=-1
    try: next(it); raise ValueError('extra physical label')
    except StopIteration: pass
    return j,tuple(right)

def allowed_output(e,left_sector,j,jmax):
    """Yield (phys_label,right_sector) valid Gauss fusion transitions for output link spin j."""
    if j<0 or j>jmax:return
    vs=phys_vertices(e)
    choices=[]
    for vi in vs:
        k=LOCAL_RANK[(vi,e)]
        if k==0:
            choices.append((j,))
        elif 1<=k<=4:
            prev=left_sector[vi]
            if prev<0:return
            choices.append(tuple(fuse(prev,j)))
        else:
            prev=left_sector[vi]
            if prev<0 or 0 not in fuse(prev,j):return
            choices.append((0,)) # implicit final value; not physical, used to build right
    for vals in product(*choices):
        after={vi:v for vi,v in zip(vs,vals)}
        rs=list(left_sector)
        for vi,v in zip(vs,vals):
            k=LOCAL_RANK[(vi,e)]
            rs[vi]=(-1 if k==5 else int(v))
        yield pack_phys(e,j,after),tuple(rs)

# Reduced recoupling primitives reconstructed from exact CG sums.
@lru_cache(None)
def rc_start(A,jo,jn,Co,Cn,r):
    num=0.0; den=0.0
    for MA in mvals(A):
      for m in mvals(jo):
        Mc=MA+m; co=cg(A,MA,jo,m,Co,Mc)
        if co==0: continue
        for a in mvals(r):
          mn=m+a
          if mn < -jn or mn > jn or ((mn+jn)&1): continue
          leaf=cg(jo,m,r,a,jn,mn)
          if leaf==0: continue
          Mcn=MA+mn; cn=cg(A,MA,jn,mn,Cn,Mcn)
          if cn==0: continue
          t=cg(Co,Mc,r,a,Cn,Mcn)
          if t: num += t*co*cn*leaf
    for Mc in mvals(Co):
      for a in mvals(r):
        Mcn=Mc+a
        if Mcn < -Cn or Mcn > Cn or ((Mcn+Cn)&1): continue
        t=cg(Co,Mc,r,a,Cn,Mcn); den += t*t
    return num/den if den else 0.0

@lru_cache(None)
def rc_prop(Ao,An,s,Co,Cn,r):
    num=0.0; den=0.0
    for Mo in mvals(Ao):
      for a in mvals(r):
        Mn=Mo+a
        if Mn < -An or Mn > An or ((Mn+An)&1): continue
        pin=cg(Ao,Mo,r,a,An,Mn)
        if pin==0: continue
        for m in mvals(s):
          Mco=Mo+m; Mcn=Mn+m
          co=cg(Ao,Mo,s,m,Co,Mco); cn=cg(An,Mn,s,m,Cn,Mcn)
          if co==0 or cn==0: continue
          tout=cg(Co,Mco,r,a,Cn,Mcn)
          if tout: num += tout*pin*co*cn
    for Mc in mvals(Co):
      for a in mvals(r):
        Mcn=Mc+a
        if Mcn < -Cn or Mcn > Cn or ((Mcn+Cn)&1): continue
        t=cg(Co,Mc,r,a,Cn,Mcn); den += t*t
    return num/den if den else 0.0

@lru_cache(None)
def rc_end(Ao,An,jo,jn,C,r):
    tr=0.0
    for Mo in mvals(Ao):
      for a in mvals(r):
        Mn=Mo+a
        if Mn < -An or Mn > An or ((Mn+An)&1): continue
        pin=cg(Ao,Mo,r,a,An,Mn)
        if pin==0: continue
        b=-a; ph=sc.eps_phase(r,a)
        for m in mvals(jo):
          mn=m+b
          if mn < -jn or mn > jn or ((mn+jn)&1): continue
          leaf=cg(jo,m,r,b,jn,mn)
          if leaf==0: continue
          M=Mo+m
          if Mn+mn!=M:continue
          co=cg(Ao,Mo,jo,m,C,M); cn=cg(An,Mn,jn,mn,C,M)
          if co and cn: tr += pin*ph*leaf*co*cn
    return tr/(C+1)

# Plaquette corner local ranges in adapted trees.
PCORNER={}
for pid,p in enumerate(gs.PLAQ):
    dd={}
    for v,e1,e2 in p['corners']:
        vi=gs.SID[v]; a=LOCAL_RANK[(vi,e1)]; b=LOCAL_RANK[(vi,e2)]
        if a>b:a,b=b,a
        dd[vi]=(a,b)
    PCORNER[pid]=dd

@lru_cache(None)
def char_coeff(r):
    return (1j**r)*(2.0*(r+1)/X)*jv(r+1,X)

def cumulative_before_after(vi,e,left,right,j):
    k=LOCAL_RANK[(vi,e)]
    before = None if k==0 else left[vi]
    after = 0 if k==5 else right[vi]
    return k,before,after

@lru_cache(maxsize=2_000_000)
def vertex_factor_cached(pid,vi,k,ob,oa,nb,na,jo,jn,r):
    # ob/nb use -2 sentinel for None
    a,b=PCORNER[pid][vi]
    obv=None if ob==-2 else ob; nbv=None if nb==-2 else nb
    if k<a:
        return 1.0 if (jo==jn and oa==na and (obv==nbv)) else 0.0
    if k==a:
        if a==0:
            # first cumulative is leaf itself; link CG is represented by active reduced line with coefficient 1
            return 1.0 if (oa==jo and na==jn and jn in fuse(jo,r)) else 0.0
        if obv!=nbv:return 0.0
        return rc_start(obv,jo,jn,oa,na,r)
    if a<k<b:
        if jo!=jn:return 0.0
        return rc_prop(obv,nbv,jo,oa,na,r)
    if k==b:
        if oa!=na:return 0.0
        return rc_end(obv,nbv,jo,jn,oa,r)
    return 1.0 if (jo==jn and oa==na and obv==nbv) else 0.0

def local_op_factor(pid,e,oldL,oldR,newL,newR,jo,jn,r):
    p=gs.PLAQ[pid]
    affected=e in p['edges']
    if affected:
        if jn not in fuse(jo,r):return 0j
        z=math.sqrt((jo+1)/(jn+1))
    else:
        if jn!=jo:return 0j
        z=1.0
    # both endpoints: if vertex is a plaquette corner use recoupled line, otherwise identity.
    for vi in ENDS[e]:
        k,ob,oa=cumulative_before_after(vi,e,oldL,oldR,jo)
        _,nb,na=cumulative_before_after(vi,e,newL,newR,jn)
        if vi in PCORNER[pid]:
            zz=vertex_factor_cached(pid,vi,k,-2 if ob is None else ob,oa,-2 if nb is None else nb,na,jo,jn,r)
            if zz==0:return 0j
            z*=zz
        else:
            if jo!=jn or oa!=na or ob!=nb:return 0j
    return complex(z)

@dataclass
class MPS:
    cores:list                 # list dict[(lb,phys,rb)] = complex
    bond_sectors:list          # length 25, list sector tuple per bond id
    scale:complex=1+0j
    error_bound:float=0.0


def vacuum_mps():
    empty=(-1,)*8
    sectors=[[empty]]; cores=[]; left=empty
    for e in ORDER:
        # j=0, all created channels 0
        opts=list(allowed_output(e,left,0,0)); assert len(opts)==1
        phys,right=opts[0]
        cores.append({(0,phys,0):1+0j}); sectors.append([right]); left=right
    assert left==empty
    return MPS(cores,sectors,1+0j,0.0)

def core_adj(core):
    d=defaultdict(list)
    for (lb,p,rb),z in core.items():d[lb].append((p,rb,z))
    return d

# sparse block compression. Matrix keys row=(outlb,phys), col=(inrb,r), grouped by right sector.
def compress_site(entries_by_sec,budget):
    # returns core dict, carry list tuples (newbond,inrb,r,sector,val), sectors list, discarded norm
    decomps=[]; svall=[]
    # each block can further decompose into connected components to keep dense SVD tiny.
    isolated=[]
    for sec,ed in entries_by_sec.items():
        # Fast path: a very common recoupled case is a weighted partial permutation,
        # i.e. every row and every column occurs once. Each component is then exactly 1x1.
        nz=[(row,col,z) for (row,col),z in ed.items() if abs(z)!=0]
        if not nz: continue
        rset={x[0] for x in nz}; cset={x[1] for x in nz}
        if len(rset)==len(nz) and len(cset)==len(nz):
            for row,col,z in nz:
                isolated.append((sec,row,col,complex(z)))
                svall.append((float(abs(z)**2),-(len(isolated)),0))
            continue
        rows={}; cols={}; triples=[]
        for row,col,z in nz:
            ri=rows.setdefault(row,len(rows)); ci=cols.setdefault(col,len(cols)); triples.append((ri,ci,z,row,col))
        nr,nc=len(rows),len(cols)
        # union find bipartite
        par=list(range(nr+nc)); sz=[1]*(nr+nc)
        def find(x):
            while par[x]!=x:
                par[x]=par[par[x]]; x=par[x]
            return x
        def union(a,b):
            a=find(a);b=find(b)
            if a==b:return
            if sz[a]<sz[b]:a,b=b,a
            par[b]=a;sz[a]+=sz[b]
        for ri,ci,*_ in triples: union(ri,nr+ci)
        comp=defaultdict(list)
        for t in triples: comp[find(t[0])].append(t)
        invr=[None]*nr; invc=[None]*nc
        for x,i in rows.items():invr[i]=x
        for x,i in cols.items():invc[i]=x
        for ts in comp.values():
            rids=sorted({t[0] for t in ts}); cids=sorted({t[1] for t in ts})
            rr={x:i for i,x in enumerate(rids)}; cc={x:i for i,x in enumerate(cids)}
            if len(rids)==1 or len(cids)==1:
                # exact rank-one vector block, no SVD needed
                A=np.zeros((len(rids),len(cids)),complex)
                for ri,ci,z,*_ in ts:A[rr[ri],cc[ci]]+=z
                s0=float(np.linalg.norm(A))
                if s0==0:continue
                U=A/s0 if len(cids)==1 else np.ones((1,1),complex)
                if len(rids)==1:
                    # A = [1] * s * (A/s)^ row
                    U=np.ones((1,1),complex); Vh=A/s0
                else:
                    Vh=np.ones((1,1),complex)
                s=np.array([s0]); rank1=True
            else:
                A=np.zeros((len(rids),len(cids)),complex)
                for ri,ci,z,*_ in ts:A[rr[ri],cc[ci]]+=z
                U,s,Vh=np.linalg.svd(A,full_matrices=False);rank1=False
            di=len(decomps)
            decomps.append((sec,rids,cids,invr,invc,U,s,Vh))
            for kk,x in enumerate(s): svall.append((float(x*x),di,kk))
    # global smallest discard under Frobenius budget. Negative component ids encode
    # isolated 1x1 blocks and avoid any SVD/union-find work.
    svall.sort(key=lambda x:x[0]); disc=set(); used=0.0; B2=budget*budget
    for sq,di,kk in svall:
        if used+sq <= B2*(1+1e-14):disc.add((di,kk));used+=sq
        else:break
    core=defaultdict(complex); carry=[]; sectors=[]
    for ii,(sec,row,col,z) in enumerate(isolated, start=1):
        if (-ii,0) in disc: continue
        bn=len(sectors); sectors.append(sec)
        a=abs(z)
        if a==0: continue
        outlb,phys=row; inrb,r=col
        core[(outlb,phys,bn)] += z/a
        carry.append((bn,inrb,r,sec,complex(a)))
    for di,(sec,rids,cids,invr,invc,U,s,Vh) in enumerate(decomps):
        for kk,x in enumerate(s):
            if (di,kk) in disc:continue
            bn=len(sectors);sectors.append(sec)
            for ii,ri in enumerate(rids):
                z=U[ii,kk]
                if abs(z)>1e-15:
                    outlb,phys=invr[ri]; core[(outlb,phys,bn)]+=z
            for jj,ci in enumerate(cids):
                z=x*Vh[kk,jj]
                if abs(z)>1e-15:
                    inrb,r=invc[ci]; carry.append((bn,inrb,r,sec,complex(z)))
    return dict(core),carry,sectors,math.sqrt(used),{'components':len(decomps)+len(isolated),'singulars':len(svall),'kept':len(sectors),'rank1_components':len(isolated)+sum(1 for d in decomps if len(d[1])==1 or len(d[2])==1),'isolated_1x1':len(isolated)}


def apply_gate(state:MPS,pid:int,jmax:int,budget:float,verbose=False):
    n=24; newcores=[]; newbondsecs=[[(-1,)*8]]
    # carry from normalized output-left basis into raw (input left,r,new left sector)
    carry=[]
    for r in range(jmax+1):
        c=cmath.exp(-1j*32*PHI)*char_coeff(r)
        if abs(c)>1e-18: carry.append((0,0,r,(-1,)*8,c))
    disc_total=0.0; profiles=[]
    input_adjs=[core_adj(c) for c in state.cores]
    # distribute budget over 23 compression cuts by equal L2 allocation; triangle-bound uses sum, so per-cut budget/23.
    percut=budget/23.0 if budget>0 else 0.0
    for si,e in enumerate(ORDER):
        oldLsecs=state.bond_sectors[si]; oldRsecs=state.bond_sectors[si+1]
        entries=defaultdict(lambda:defaultdict(complex))
        # index carry by inlb
        cb=defaultdict(list)
        for outlb,inlb,r,nls,z in carry:cb[inlb].append((outlb,r,nls,z))
        cache_out={}
        for inlb,carrs in cb.items():
            for oldphys,inrb,az in input_adjs[si].get(inlb,()):
                oldL=oldLsecs[inlb]; oldR=oldRsecs[inrb]; jo=int(oldphys[0])
                for outlb,r,nls,cz in carrs:
                    # link output spins
                    js=fuse(jo,r) if e in gs.PLAQ[pid]['edges'] else (jo,)
                    for jn in js:
                        if jn>jmax:continue
                        key=(nls,jn)
                        opts=cache_out.get(key)
                        if opts is None:
                            opts=list(allowed_output(e,nls,jn,jmax));cache_out[key]=opts
                        for nphys,nrs in opts:
                            f=local_op_factor(pid,e,oldL,oldR,nls,nrs,jo,jn,r)
                            if abs(f)<1e-16:continue
                            entries[nrs][((outlb,nphys),(inrb,r))] += cz*az*f
        if si==n-1:
            # exact terminal closure: input right terminal is unique, operator irrep line is coherently summed.
            empty=(-1,)*8; ed=entries.get(empty,{})
            last=defaultdict(complex)
            for (row,col),z in ed.items():
                outlb,nphys=row; inrb,r=col
                if inrb!=0: raise RuntimeError('nonterminal input bond at final site')
                last[(outlb,nphys,0)] += z
            if not last: raise RuntimeError('empty terminal gate state')
            # Previous cores are left-isometric, so final-core Euclidean norm is global state norm.
            norm=math.sqrt(sum(abs(z)**2 for z in last.values()))
            if norm==0:raise RuntimeError('zero final norm')
            for k in list(last):last[k]/=norm
            newcores.append(dict(last));newbondsecs.append([empty])
            profiles.append({'site':si,'edge':e,'terminal_entries':len(ed),'terminal_core_nnz':len(last),'terminal_rank1_direct':True,'norm_before_rescale':norm})
            scale=state.scale*norm
            if verbose:print('site',si,'e',e,'TERMINAL',len(ed),len(last),'norm',norm,flush=True)
            return MPS(newcores,newbondsecs,scale,state.error_bound+disc_total),{'pid':pid,'jmax':jmax,'budget':budget,'discarded_l2':disc_total,'profiles':profiles,'scale_abs':abs(scale),'max_bond':max(len(s) for s in newbondsecs)}
        core,carry,newsecs,disc,prof=compress_site(entries,percut)
        disc_total += disc
        newcores.append(core);newbondsecs.append(newsecs)
        prof.update({'site':si,'edge':e,'sectors':len(set(newsecs)),'bond':len(newsecs),'core_nnz':len(core),'carry_nnz':len(carry),'discarded_l2':disc})
        profiles.append(prof)
        if verbose:print('site',si,'e',e,'bond',len(newsecs),'sec',prof['sectors'],'comp',prof['components'],'r1',prof['rank1_components'],'nnz',len(core),len(carry),'disc',disc,flush=True)
    raise RuntimeError('unreachable')

def expand_mps(state:MPS,max_states=1000000):
    # Exact expansion for validation only. physical configuration tuple over 24 grouped sites.
    cur={(0,()):1+0j}
    for si,core in enumerate(state.cores):
        adj=core_adj(core); nxt=defaultdict(complex)
        for (lb,prefix),a in cur.items():
            for phys,rb,z in adj.get(lb,()):
                nxt[(rb,prefix+(phys,))]+=a*z
        if len(nxt)>max_states:raise RuntimeError(f'expansion {len(nxt)} > max')
        cur=nxt
    return {p:state.scale*a for (rb,p),a in cur.items() if rb==0}

def direct_adapted_matrix_element(old_phys,new_phys,pid,r):
    # reconstruct per-vertex adapted spins and paths from grouped physical configs and multiply reference corner DP.
    def decode(config):
        ls=[None]*24; channels={vi:[] for vi in range(8)}; left=(-1,)*8
        # path arrays ordered: cumulative after leaf2..leaf5; final0
        ch={vi:{} for vi in range(8)}
        for si,e in enumerate(ORDER):
            phys=config[si];j=int(phys[0]);ls[e]=j
            it=iter(phys[1:])
            for vi in phys_vertices(e):
                k=LOCAL_RANK[(vi,e)]
                if 1<=k<=4:ch[vi][k]=int(next(it))
        paths={vi:tuple(ch[vi][k] for k in range(1,5))+(0,) for vi in range(8)}
        return tuple(ls),paths
    ol,op=decode(old_phys);nl,npth=decode(new_phys);p=gs.PLAQ[pid]
    if any(ol[e]!=nl[e] for e in range(24) if e not in p['edges']):return 0j
    z=1+0j
    for e in p['edges']:
        if nl[e] not in fuse(ol[e],r):return 0j
        z*=math.sqrt((ol[e]+1)/(nl[e]+1))
    for v,e1,e2 in p['corners']:
        vi=gs.SID[v]; inc=INC_ORDER[vi]
        osp=tuple(ol[e] for e in inc); nsp=tuple(nl[e] for e in inc)
        i=LOCAL_RANK[(vi,e1)];j=LOCAL_RANK[(vi,e2)]
        z*=sc.corner_element_dp(osp,op[vi],nsp,npth[vi],i,j,r)
    return z

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,default=0);ap.add_argument('--jmax',type=int,default=4);ap.add_argument('--budget',type=float,default=0);ap.add_argument('--verbose',action='store_true');args=ap.parse_args()
    v=vacuum_mps();v,rep=apply_gate(v,args.pid,args.jmax,args.budget,args.verbose);print(json.dumps(rep,indent=2))
    if args.jmax<=6:
      ex=expand_mps(v,100000)
      print('expanded',len(ex),'norm',math.sqrt(sum(abs(a)**2 for a in ex.values())))
      # vacuum -> states should be one per r and amplitudes phase*coeff(r), up to basis phase. print.
      for x,a in list(ex.items())[:20]:print(a,x)

# ---------------- Full-state right-canonical compression ----------------
def _right_factor_blocks(entries_by_sec,budget):
    """Factor A[old_left, (phys,right)] blockwise by conserved left sector.
    Returns transform U*S from old_left to new_left and right-isometric core Vh.
    """
    decomps=[]; isolated=[]; svall=[]
    for sec,ed in entries_by_sec.items():
        nz=[(row,col,z) for (row,col),z in ed.items() if abs(z)!=0]
        if not nz:continue
        rset={x[0] for x in nz}; cset={x[1] for x in nz}
        if len(rset)==len(nz) and len(cset)==len(nz):
            for row,col,z in nz:
                isolated.append((sec,row,col,complex(z))); svall.append((float(abs(z)**2),-len(isolated),0))
            continue
        rows={};cols={};tri=[]
        for row,col,z in nz:
            ri=rows.setdefault(row,len(rows));ci=cols.setdefault(col,len(cols));tri.append((ri,ci,z))
        nr,nc=len(rows),len(cols);par=list(range(nr+nc));sz=[1]*(nr+nc)
        def find(x):
            while par[x]!=x:par[x]=par[par[x]];x=par[x]
            return x
        def union(a,b):
            a=find(a);b=find(b)
            if a==b:return
            if sz[a]<sz[b]:a,b=b,a
            par[b]=a;sz[a]+=sz[b]
        for ri,ci,_ in tri:union(ri,nr+ci)
        comps=defaultdict(list)
        for ri,ci,z in tri:comps[find(ri)].append((ri,ci,z))
        invr=[None]*nr;invc=[None]*nc
        for x,i in rows.items():invr[i]=x
        for x,i in cols.items():invc[i]=x
        for ts in comps.values():
            rids=sorted({x[0] for x in ts});cids=sorted({x[1] for x in ts})
            rr={x:i for i,x in enumerate(rids)};cc={x:i for i,x in enumerate(cids)}
            A=np.zeros((len(rids),len(cids)),complex)
            for ri,ci,z in ts:A[rr[ri],cc[ci]]+=z
            if len(rids)==1:
                s0=float(np.linalg.norm(A));
                if s0==0:continue
                U=np.ones((1,1),complex);s=np.array([s0]);Vh=A/s0
            elif len(cids)==1:
                s0=float(np.linalg.norm(A));
                if s0==0:continue
                U=A/s0;s=np.array([s0]);Vh=np.ones((1,1),complex)
            else:
                U,s,Vh=np.linalg.svd(A,full_matrices=False)
            di=len(decomps);decomps.append((sec,rids,cids,invr,invc,U,s,Vh))
            for k,x in enumerate(s):svall.append((float(x*x),di,k))
    svall.sort(key=lambda x:x[0]);disc=set();used=0.0;B2=budget*budget
    for sq,di,k in svall:
        if used+sq<=B2*(1+1e-14):disc.add((di,k));used+=sq
        else:break
    transform=defaultdict(complex) # (oldlb,newlb)
    core=defaultdict(complex)      # (newlb,phys,rb)
    sectors=[]
    for ii,(sec,row,col,z) in enumerate(isolated,start=1):
        if (-ii,0) in disc:continue
        bn=len(sectors);sectors.append(sec);a=abs(z)
        if a==0:continue
        # choose U phase so core is unit right vector 1; transform carries z.
        transform[(row,bn)]+=z
        phys,rb=col;core[(bn,phys,rb)]+=1.0
    for di,(sec,rids,cids,invr,invc,U,s,Vh) in enumerate(decomps):
        for k,x in enumerate(s):
            if (di,k) in disc:continue
            bn=len(sectors);sectors.append(sec)
            for ii,ri in enumerate(rids):
                z=U[ii,k]*x
                if abs(z)>1e-15:transform[(invr[ri],bn)]+=z
            for jj,ci in enumerate(cids):
                z=Vh[k,jj]
                if abs(z)>1e-15:
                    phys,rb=invc[ci];core[(bn,phys,rb)]+=z
    info={'components':len(isolated)+len(decomps),'isolated_1x1':len(isolated),'singulars':len(svall),'kept':len(sectors)}
    return dict(transform),dict(core),sectors,math.sqrt(used),info

def right_compress(state:MPS,budget:float,verbose=False):
    cores=[dict(c) for c in state.cores]; bsecs=[list(x) for x in state.bond_sectors]
    percut=budget/23.0 if budget>0 else 0.0; total=0.0;profiles=[]
    for si in range(23,0,-1):
        # core si grouped by left sector
        by=defaultdict(lambda:defaultdict(complex))
        for (lb,phys,rb),z in cores[si].items():
            sec=bsecs[si][lb];by[sec][(lb,(phys,rb))]+=z
        T,newcore,newsecs,disc,info=_right_factor_blocks(by,percut);total+=disc
        # propagate T into previous core right index
        tadj=defaultdict(list)
        for (old,new),z in T.items():tadj[old].append((new,z))
        prev=defaultdict(complex)
        for (lb,phys,oldrb),a in cores[si-1].items():
            for newrb,z in tadj.get(oldrb,()):prev[(lb,phys,newrb)]+=a*z
        cores[si-1]=dict(prev);cores[si]=newcore;bsecs[si]=newsecs
        info.update({'site':si,'bond':len(newsecs),'core_nnz':len(newcore),'prev_nnz':len(prev),'discarded_l2':disc});profiles.append(info)
        if verbose:print('R',si,'bond',len(newsecs),'comp',info['components'],'iso',info['isolated_1x1'],'nnz',len(newcore),'disc',disc,flush=True)
    # Right suffix is isometric; norm resides in site0 tensor.
    norm=math.sqrt(sum(abs(z)**2 for z in cores[0].values()))
    if norm==0:raise RuntimeError('zero norm after right compression')
    for k in list(cores[0]):cores[0][k]/=norm
    scale=state.scale*norm
    out=MPS(cores,bsecs,scale,state.error_bound+total)
    return out,{'budget':budget,'discarded_l2':total,'profiles':profiles,'norm_before_rescale':norm,'scale_abs':abs(scale),'max_bond':max(len(s) for s in bsecs)}

def overlap(a:MPS,b:MPS):
    """Physical inner product including scales. Requires same site basis/order."""
    env={(0,0):1+0j}
    for ca,cb in zip(a.cores,b.cores):
        aa=defaultdict(list);bb=defaultdict(list)
        for (l,p,r),z in ca.items():aa[(l,p)].append((r,z))
        for (l,p,r),z in cb.items():bb[(l,p)].append((r,z))
        nxt=defaultdict(complex)
        # iterate common (leftA,leftB), then common phys by indexing separately
        byA=defaultdict(lambda:defaultdict(list));byB=defaultdict(lambda:defaultdict(list))
        for (l,p),xs in aa.items():byA[l][p]=xs
        for (l,p),xs in bb.items():byB[l][p]=xs
        for (la,lb),e in env.items():
            da=byA.get(la,{});db=byB.get(lb,{})
            for p in da.keys() & db.keys():
                for ra,za in da[p]:
                    for rb,zb in db[p]:nxt[(ra,rb)]+=e*np.conjugate(za)*zb
        env=nxt
    return np.conjugate(a.scale)*b.scale*env.get((0,0),0j)

# ---------------- Minimal-support plaquette MPO application ----------------
def gate_relevant_sites(pid:int):
    """Global sweep positions where the reduced plaquette line r can affect a local tensor."""
    rel=set()
    p=gs.PLAQ[pid]
    pedges=set(p['edges'])
    for si,e in enumerate(ORDER):
        if e in pedges:
            rel.add(si)
        for vi in ENDS[e]:
            if vi in PCORNER[pid]:
                k=LOCAL_RANK[(vi,e)]; a,b=PCORNER[pid][vi]
                if a <= k <= b:
                    rel.add(si)
    if not rel: raise RuntimeError(('empty plaquette support',pid))
    return tuple(sorted(rel))

def apply_gate_span(state:MPS,pid:int,jmax:int,budget:float=0.0,verbose=False):
    """Apply one plaquette gate with the character irrep line present only on its exact reduced support span.

    Exact outside [start,end]: prefix and suffix MPS tensors are reused verbatim. At `start`
    the global character irrep r is injected; at `end` it is coherently summed and the bond is
    reattached directly to the input suffix. This is algebraically identical to `apply_gate` but
    removes the pure-gauge r multiplicity outside the plaquette's recoupling support.
    """
    rel=gate_relevant_sites(pid); start,end=rel[0],rel[-1]
    n=24
    # Exact untouched prefix.
    newcores=[dict(c) for c in state.cores[:start]]
    newbondsecs=[list(s) for s in state.bond_sectors[:start+1]]
    leftsecs=state.bond_sectors[start]
    carry=[]
    for lb,sec in enumerate(leftsecs):
        for r in range(jmax+1):
            c=cmath.exp(-1j*32*PHI)*char_coeff(r)
            if abs(c)>1e-18: carry.append((lb,lb,r,sec,c))
    input_adjs=[core_adj(c) for c in state.cores]
    disc_total=0.0; profiles=[]
    cuts=max(1,end-start)
    percut=budget/cuts if budget>0 else 0.0
    for si in range(start,end+1):
        e=ORDER[si]; oldLsecs=state.bond_sectors[si]; oldRsecs=state.bond_sectors[si+1]
        entries=defaultdict(lambda:defaultdict(complex));cb=defaultdict(list)
        for outlb,inlb,r,nls,z in carry: cb[inlb].append((outlb,r,nls,z))
        cache_out={}
        for inlb,carrs in cb.items():
            for oldphys,inrb,az in input_adjs[si].get(inlb,()):
                oldL=oldLsecs[inlb]; oldR=oldRsecs[inrb]; jo=int(oldphys[0])
                for outlb,r,nls,cz in carrs:
                    js=fuse(jo,r) if e in gs.PLAQ[pid]['edges'] else (jo,)
                    for jn in js:
                        if jn>jmax:continue
                        key=(nls,jn); opts=cache_out.get(key)
                        if opts is None:
                            opts=list(allowed_output(e,nls,jn,jmax));cache_out[key]=opts
                        for nphys,nrs in opts:
                            f=local_op_factor(pid,e,oldL,oldR,nls,nrs,jo,jn,r)
                            if abs(f)<1e-16:continue
                            entries[nrs][((outlb,nphys),(inrb,r))] += cz*az*f
        if si==end:
            # Coherently close r and reconnect to the unchanged input suffix bond.
            core=defaultdict(complex);bad=0;terms=0
            for nrs,ed in entries.items():
                for (row,col),z in ed.items():
                    outlb,nphys=row; inrb,r=col; terms+=1
                    if nrs != oldRsecs[inrb]:
                        if abs(z)>1e-13: bad+=1
                        continue
                    core[(outlb,nphys,inrb)] += z
            if bad: raise RuntimeError(('span closure sector mismatch',pid,si,bad))
            newcores.append(dict(core)); newbondsecs.append(list(oldRsecs))
            profiles.append({'site':si,'edge':e,'span_terminal':True,'terms':terms,'core_nnz':len(core),'bond':len(oldRsecs)})
            if verbose: print('SPAN CLOSE',pid,'site',si,'e',e,'terms',terms,'core',len(core),'bond',len(oldRsecs),flush=True)
            break
        core,carry,newsecs,disc,prof=compress_site(entries,percut);disc_total+=disc
        newcores.append(core);newbondsecs.append(newsecs)
        prof.update({'site':si,'edge':e,'bond':len(newsecs),'core_nnz':len(core),'carry_nnz':len(carry),'discarded_l2':disc})
        profiles.append(prof)
        if verbose: print('SPAN',pid,'site',si,'e',e,'bond',len(newsecs),'nnz',len(core),len(carry),'disc',disc,flush=True)
    # Exact untouched suffix after bond end+1.
    newcores.extend(dict(c) for c in state.cores[end+1:])
    newbondsecs.extend(list(s) for s in state.bond_sectors[end+2:])
    assert len(newcores)==24 and len(newbondsecs)==25
    return MPS(newcores,newbondsecs,state.scale,state.error_bound+disc_total),{
        'pid':pid,'jmax':jmax,'budget':budget,'support_sites':list(rel),'span':[start,end],
        'discarded_l2':disc_total,'profiles':profiles,'scale_abs':abs(state.scale),
        'max_bond':max(len(s) for s in newbondsecs)}

# ---------------- Exact pruned minimal-support gate ----------------
def pruned_local_transitions(pid,e,oldL,oldR,nls,oldphys,r,jmax):
    """Yield (newphys,newR,factor) without enumerating channels that local_op_factor would reject.
    Algebraically equivalent to allowed_output + local_op_factor filtering.
    """
    jo=int(oldphys[0]); p=gs.PLAQ[pid]; affected=e in p['edges']
    js=fuse(jo,r) if affected else (jo,)
    for jn in js:
        if jn>jmax: continue
        base=math.sqrt((jo+1)/(jn+1)) if affected else 1.0
        # endpoint choices contain (na, vertex_factor); na is cumulative after this leaf (0 at final leaf)
        allc=[]; ok=True
        for vi in ENDS[e]:
            k=LOCAL_RANK[(vi,e)]
            ob=None if k==0 else oldL[vi]
            oa=0 if k==5 else oldR[vi]
            nb=None if k==0 else nls[vi]
            active=False; a=b=-1
            if vi in PCORNER[pid]:
                a,b=PCORNER[pid][vi]; active=(a<=k<=b)
            if not active:
                # exact identity at this endpoint
                if jo!=jn or ob!=nb:
                    ok=False; break
                na=oa
                if k==0:
                    if na!=jn: ok=False;break
                elif 1<=k<=4:
                    if na not in fuse(nb,jn): ok=False;break
                else:
                    if 0 not in fuse(nb,jn): ok=False;break
                allc.append(((na,1.0),)); continue
            # active corner interval
            if k==a:
                if a==0:
                    # after is the leaf itself
                    na=jn
                    if jn not in fuse(jo,r): ok=False;break
                    allc.append(((na,1.0),))
                else:
                    if ob!=nb: ok=False;break
                    vals=[]
                    for na in fuse(nb,jn):
                        z=rc_start(ob,jo,jn,oa,na,r)
                        if abs(z)>=1e-16: vals.append((na,z))
                    if not vals:ok=False;break
                    allc.append(tuple(vals))
            elif a<k<b:
                if jo!=jn: ok=False;break
                vals=[]
                for na in fuse(nb,jn):
                    z=rc_prop(ob,nb,jo,oa,na,r)
                    if abs(z)>=1e-16: vals.append((na,z))
                if not vals:ok=False;break
                allc.append(tuple(vals))
            elif k==b:
                na=oa
                if k<=4:
                    if na not in fuse(nb,jn): ok=False;break
                else:
                    if 0 not in fuse(nb,jn): ok=False;break
                z=rc_end(ob,nb,jo,jn,oa,r)
                if abs(z)<1e-16:ok=False;break
                allc.append(((na,z),))
            else:
                raise RuntimeError('active logic')
        if not ok: continue
        for vals in product(*allc):
            after={vi:int(x[0]) for vi,x in zip(ENDS[e],vals)}
            nrs=list(nls); z=complex(base)
            for vi,(na,vf) in zip(ENDS[e],vals):
                k=LOCAL_RANK[(vi,e)]; nrs[vi]=(-1 if k==5 else int(na));z*=vf
            yield pack_phys(e,jn,after),tuple(nrs),z

def apply_gate_span_pruned(state:MPS,pid:int,jmax:int,budget:float=0.0,verbose=False):
    rel=gate_relevant_sites(pid);start,end=rel[0],rel[-1]
    newcores=[dict(c) for c in state.cores[:start]]
    newbondsecs=[list(s) for s in state.bond_sectors[:start+1]]
    leftsecs=state.bond_sectors[start]; carry=[]
    for lb,sec in enumerate(leftsecs):
        for r in range(jmax+1):
            c=cmath.exp(-1j*32*PHI)*char_coeff(r)
            if abs(c)>1e-18: carry.append((lb,lb,r,sec,c))
    input_adjs=[core_adj(c) for c in state.cores];disc_total=0.0;profiles=[]
    cuts=max(1,end-start);percut=budget/cuts if budget>0 else 0.0
    for si in range(start,end+1):
        e=ORDER[si];oldLsecs=state.bond_sectors[si];oldRsecs=state.bond_sectors[si+1]
        entries=defaultdict(lambda:defaultdict(complex));cb=defaultdict(list)
        for outlb,inlb,r,nls,z in carry:cb[inlb].append((outlb,r,nls,z))
        generated=0
        for inlb,carrs in cb.items():
            for oldphys,inrb,az in input_adjs[si].get(inlb,()):
                oldL=oldLsecs[inlb];oldR=oldRsecs[inrb]
                for outlb,r,nls,cz in carrs:
                    for nphys,nrs,f in pruned_local_transitions(pid,e,oldL,oldR,nls,oldphys,r,jmax):
                        generated+=1;entries[nrs][((outlb,nphys),(inrb,r))]+=cz*az*f
        if si==end:
            core=defaultdict(complex);bad=0;terms=0
            for nrs,ed in entries.items():
                for (row,col),z in ed.items():
                    outlb,nphys=row;inrb,r=col;terms+=1
                    if nrs!=oldRsecs[inrb]:
                        if abs(z)>1e-13:bad+=1
                        continue
                    core[(outlb,nphys,inrb)]+=z
            if bad:raise RuntimeError(('pruned span closure sector mismatch',pid,si,bad))
            newcores.append(dict(core));newbondsecs.append(list(oldRsecs))
            profiles.append({'site':si,'edge':e,'span_terminal':True,'generated':generated,'terms':terms,'core_nnz':len(core),'bond':len(oldRsecs)})
            break
        core,carry,newsecs,disc,prof=compress_site(entries,percut);disc_total+=disc
        newcores.append(core);newbondsecs.append(newsecs)
        prof.update({'site':si,'edge':e,'generated':generated,'bond':len(newsecs),'core_nnz':len(core),'carry_nnz':len(carry),'discarded_l2':disc});profiles.append(prof)
        if verbose:print('PRUNED',pid,'site',si,'e',e,'gen',generated,'bond',len(newsecs),'nnz',len(core),len(carry),'disc',disc,flush=True)
    newcores.extend(dict(c) for c in state.cores[end+1:]);newbondsecs.extend(list(s) for s in state.bond_sectors[end+2:])
    assert len(newcores)==24 and len(newbondsecs)==25
    return MPS(newcores,newbondsecs,state.scale,state.error_bound+disc_total),{'pid':pid,'jmax':jmax,'budget':budget,'support_sites':list(rel),'span':[start,end],'discarded_l2':disc_total,'profiles':profiles,'scale_abs':abs(state.scale),'max_bond':max(len(s) for s in newbondsecs)}

# ---------------- Closed-form Racah recoupling (exactly equivalent to CG sums) ----------------
@lru_cache(None)
def wigner6j_fast(a,b,c,d,e,f):
    """Wigner 6j in doubled-spin integer convention, via Racah formula."""
    def tri(x,y,z):
        if x<0 or y<0 or z<0 or z<abs(x-y) or z>x+y or ((x+y+z)&1): return False
        return True
    if not (tri(a,b,c) and tri(a,e,f) and tri(d,b,f) and tri(d,e,c)):
        return 0.0
    def delta(x,y,z):
        p=(x+y-z)//2; q=(x-y+z)//2; r=(-x+y+z)//2; s=(x+y+z)//2+1
        return math.sqrt(math.factorial(p)*math.factorial(q)*math.factorial(r)/math.factorial(s))
    A1=(a+b+c)//2; A2=(a+e+f)//2; A3=(d+b+f)//2; A4=(d+e+c)//2
    B1=(a+b+d+e)//2; B2=(a+c+d+f)//2; B3=(b+c+e+f)//2
    lo=max(A1,A2,A3,A4); hi=min(B1,B2,B3)
    if lo>hi:return 0.0
    sm=0.0
    for z in range(lo,hi+1):
        den=(math.factorial(z-A1)*math.factorial(z-A2)*math.factorial(z-A3)*math.factorial(z-A4)*
             math.factorial(B1-z)*math.factorial(B2-z)*math.factorial(B3-z))
        term=math.factorial(z+1)/den
        sm += -term if (z&1) else term
    return delta(a,b,c)*delta(a,e,f)*delta(d,b,f)*delta(d,e,c)*sm

# Preserve explicit-sum references for regression.
rc_start_cgsum=rc_start; rc_prop_cgsum=rc_prop; rc_end_cgsum=rc_end

@lru_cache(None)
def rc_prop(Ao,An,s,Co,Cn,r):
    # (-1)^((An+s+Co+r)/2) sqrt((An+1)(Co+1)) {Ao r An; Cn s Co}
    if (An+s+Co+r)&1:return 0.0
    ph=-1.0 if (((An+s+Co+r)//2)&1) else 1.0
    return ph*math.sqrt((An+1)*(Co+1))*wigner6j_fast(Ao,r,An,Cn,s,Co)

@lru_cache(None)
def rc_start(A,jo,jn,Co,Cn,r):
    # (-1)^((A+jo+Cn+r)/2) sqrt((jn+1)(Co+1)) {jo r jn; Cn A Co}
    if (A+jo+Cn+r)&1:return 0.0
    ph=-1.0 if (((A+jo+Cn+r)//2)&1) else 1.0
    return ph*math.sqrt((jn+1)*(Co+1))*wigner6j_fast(jo,r,jn,Cn,A,Co)

@lru_cache(None)
def rc_end(Ao,An,jo,jn,C,r):
    # (-1)^((Ao+jn+C+r)/2) sqrt((An+1)(jn+1)) {Ao r An; jn C jo}
    if (Ao+jn+C+r)&1:return 0.0
    ph=-1.0 if (((Ao+jn+C+r)//2)&1) else 1.0
    return ph*math.sqrt((An+1)*(jn+1))*wigner6j_fast(Ao,r,An,jn,C,jo)

# ---------------- Cached local transition templates ----------------
@lru_cache(maxsize=2_000_000)
def _local_transition_template(pid,e,jo,r,edata):
    """Return tuples (jn,na0,na1,factor) depending only on local endpoint channels.
    edata is ((vi,k,ob_s,oa,nb_s),(...)); sentinel -2 means None.
    """
    p=gs.PLAQ[pid];affected=e in p['edges'];out=[]
    for jn in (fuse(jo,r) if affected else (jo,)):
        base=math.sqrt((jo+1)/(jn+1)) if affected else 1.0
        allc=[];ok=True
        for vi,k,obs,oa,nbs in edata:
            ob=None if obs==-2 else obs;nb=None if nbs==-2 else nbs
            active=False;a=b=-1
            if vi in PCORNER[pid]:a,b=PCORNER[pid][vi];active=(a<=k<=b)
            if not active:
                if jo!=jn or ob!=nb:ok=False;break
                na=oa
                if k==0:
                    if na!=jn:ok=False;break
                elif 1<=k<=4:
                    if na not in fuse(nb,jn):ok=False;break
                else:
                    if 0 not in fuse(nb,jn):ok=False;break
                allc.append(((na,1.0),));continue
            if k==a:
                if a==0:
                    na=jn
                    if jn not in fuse(jo,r):ok=False;break
                    allc.append(((na,1.0),))
                else:
                    if ob!=nb:ok=False;break
                    vals=tuple((na,rc_start(ob,jo,jn,oa,na,r)) for na in fuse(nb,jn))
                    vals=tuple((na,z) for na,z in vals if abs(z)>=1e-16)
                    if not vals:ok=False;break
                    allc.append(vals)
            elif a<k<b:
                if jo!=jn:ok=False;break
                vals=tuple((na,rc_prop(ob,nb,jo,oa,na,r)) for na in fuse(nb,jn))
                vals=tuple((na,z) for na,z in vals if abs(z)>=1e-16)
                if not vals:ok=False;break
                allc.append(vals)
            elif k==b:
                na=oa
                if k<=4:
                    if na not in fuse(nb,jn):ok=False;break
                else:
                    if 0 not in fuse(nb,jn):ok=False;break
                z=rc_end(ob,nb,jo,jn,oa,r)
                if abs(z)<1e-16:ok=False;break
                allc.append(((na,z),))
        if not ok:continue
        for vals in product(*allc):
            z=complex(base)
            for _,vf in vals:z*=vf
            out.append((int(jn),int(vals[0][0]),int(vals[1][0]),z))
    return tuple(out)

def pruned_local_transitions(pid,e,oldL,oldR,nls,oldphys,r,jmax):
    jo=int(oldphys[0]); ed=[]
    for vi in ENDS[e]:
        k=LOCAL_RANK[(vi,e)];ob=None if k==0 else oldL[vi];oa=0 if k==5 else oldR[vi];nb=None if k==0 else nls[vi]
        ed.append((vi,k,-2 if ob is None else int(ob),int(oa),-2 if nb is None else int(nb)))
    for jn,na0,na1,z in _local_transition_template(pid,e,jo,r,tuple(ed)):
        if jn>jmax:continue
        after={ENDS[e][0]:na0,ENDS[e][1]:na1};nrs=list(nls)
        for vi,na in after.items():
            k=LOCAL_RANK[(vi,e)];nrs[vi]=(-1 if k==5 else int(na))
        yield pack_phys(e,jn,after),tuple(nrs),z

# ---------------- Exact MPO for H_B = sum_p chi_{1/2}(U_p) ----------------
_HB_SPAN={p:(gate_relevant_sites(p)[0],gate_relevant_sites(p)[-1]) for p in range(24)}
_HB_START=defaultdict(list)
for _p,(_a,_b) in _HB_SPAN.items(): _HB_START[_a].append(_p)
HB_IDLE=-1; HB_DONE=-2

def apply_HB(state:MPS,jmax:int,budget:float=0.0,verbose=False):
    """Apply H_B=sum_p chi_{1/2}(U_p) exactly (up to declared MPS compression budget).
    MPO automaton tracks only one active plaquette term; fixed character r=1.
    """
    n=24;newcores=[];newbondsecs=[[(-1,)*8]]
    # carry = (outlb,inlb,mode,new_left_sector,z), mode IDLE/DONE/pid
    carry=[(0,0,HB_IDLE,(-1,)*8,1+0j)]
    input_adjs=[core_adj(c) for c in state.cores];disc_total=0.0;profiles=[]
    percut=budget/23.0 if budget>0 else 0.0
    for si,e in enumerate(ORDER):
        oldLsecs=state.bond_sectors[si];oldRsecs=state.bond_sectors[si+1]
        entries=defaultdict(lambda:defaultdict(complex));cb=defaultdict(list);gen=0
        for outlb,inlb,mode,nls,z in carry:cb[inlb].append((outlb,mode,nls,z))
        for inlb,carrs in cb.items():
          for oldphys,inrb,az in input_adjs[si].get(inlb,()):
            oldL=oldLsecs[inlb];oldR=oldRsecs[inrb]
            for outlb,mode,nls,cz in carrs:
              # Identity propagation helper: only valid when output left sector matches old left.
              if mode==HB_IDLE or mode==HB_DONE:
                if nls==oldL:
                  # IDLE->IDLE or DONE->DONE
                  entries[oldR][((outlb,oldphys),(inrb,mode))]+=cz*az;gen+=1
                # IDLE may start any plaquette whose minimal support begins here.
                if mode==HB_IDLE:
                  for pid in _HB_START.get(si,()):
                    for nphys,nrs,f in pruned_local_transitions(pid,e,oldL,oldR,nls,oldphys,1,jmax):
                      nm=HB_DONE if si==_HB_SPAN[pid][1] else pid
                      entries[nrs][((outlb,nphys),(inrb,nm))]+=cz*az*f;gen+=1
              else:
                pid=mode;a,b=_HB_SPAN[pid]
                for nphys,nrs,f in pruned_local_transitions(pid,e,oldL,oldR,nls,oldphys,1,jmax):
                  nm=HB_DONE if si==b else pid
                  entries[nrs][((outlb,nphys),(inrb,nm))]+=cz*az*f;gen+=1
        if si==n-1:
            empty=(-1,)*8;ed=entries.get(empty,{})
            last=defaultdict(complex);accepted=0
            for (row,col),z in ed.items():
                outlb,nphys=row;inrb,mode=col
                if inrb!=0:continue
                if mode!=HB_DONE:continue
                last[(outlb,nphys,0)]+=z;accepted+=1
            if not last:raise RuntimeError('empty HB terminal')
            norm=math.sqrt(sum(abs(z)**2 for z in last.values()))
            if norm==0:raise RuntimeError('zero HB norm')
            for k in list(last):last[k]/=norm
            newcores.append(dict(last));newbondsecs.append([empty])
            profiles.append({'site':si,'edge':e,'terminal':True,'generated':gen,'accepted':accepted,'core_nnz':len(last),'norm_before_rescale':norm})
            return MPS(newcores,newbondsecs,state.scale*norm,state.error_bound+disc_total),{'operator':'HB','jmax':jmax,'budget':budget,'discarded_l2':disc_total,'profiles':profiles,'scale_abs':abs(state.scale*norm),'max_bond':max(map(len,newbondsecs))}
        core,carry0,newsecs,disc,prof=compress_site(entries,percut);disc_total+=disc
        # compress_site returns (newbond,inrb,mode,sector,val) already semantically compatible
        carry=[(bn,inrb,mode,sec,z) for bn,inrb,mode,sec,z in carry0]
        newcores.append(core);newbondsecs.append(newsecs)
        prof.update({'site':si,'edge':e,'generated':gen,'bond':len(newsecs),'core_nnz':len(core),'carry_nnz':len(carry),'discarded_l2':disc});profiles.append(prof)
        if verbose:print('HB',si,e,'gen',gen,'bond',len(newsecs),'carry',len(carry),'disc',disc,flush=True)
    raise RuntimeError('unreachable HB')

# ---------------- MPS linear algebra for Krylov ----------------
def mps_linear_combination(states,coeffs):
    """Exact direct-sum MPS for sum_i coeff_i * |state_i>. Common physical basis/order required."""
    assert len(states)==len(coeffs) and states
    if len(states)==1:
        s=states[0]; fac=complex(coeffs[0])*s.scale
        cores=[dict(c) for c in s.cores]
        for k in list(cores[0]): cores[0][k]*=fac
        return MPS(cores,[list(x) for x in s.bond_sectors],1+0j,abs(coeffs[0])*s.error_bound)
    ns=len(states); n=24
    # Internal bond offsets per state, per bond.
    offsets=[[0]*ns for _ in range(n+1)]
    bsecs=[[(-1,)*8]]
    for b in range(1,n):
        arr=[];off=0
        for i,s in enumerate(states):
            offsets[b][i]=off; arr.extend(s.bond_sectors[b]);off+=len(s.bond_sectors[b])
        bsecs.append(arr)
    bsecs.append([[(-1,)*8][0]])
    cores=[]
    for si in range(n):
        out=defaultdict(complex)
        for i,(s,c) in enumerate(zip(states,coeffs)):
            fac=complex(c)*s.scale if si==0 else 1+0j
            lo=offsets[si][i] if si>0 else 0
            ro=offsets[si+1][i] if si+1<n else 0
            for (lb,p,rb),z in s.cores[si].items():
                out[(lb+lo,p,rb+ro)] += fac*z
        cores.append(dict(out))
    err=sum(abs(complex(c))*s.error_bound for s,c in zip(states,coeffs))
    return MPS(cores,bsecs,1+0j,err)

def normalized_copy(state:MPS):
    """Normalize using exact overlap; absorb normalization into global scale only."""
    n2=float(np.real(overlap(state,state)))
    if n2<=0:raise RuntimeError(('nonpositive norm',n2))
    n=math.sqrt(n2)
    return MPS([dict(c) for c in state.cores],[list(x) for x in state.bond_sectors],state.scale/n,state.error_bound/n),n

def hb_lanczos_mps(m:int,jmax:int,compression_budget:float=0.0,verbose=False):
    """Lanczos tridiagonalization of H_B from vacuum using the exact HB MPO."""
    q=vacuum_mps(); q,_=normalized_copy(q); qprev=None; beta_prev=0.0
    al=[];be=[]; qs=[q]; reports=[]
    for k in range(m):
        hq,hr=apply_HB(q,jmax,compression_budget,False)
        alpha=overlap(q,hq); alpha=float(alpha.real)
        parts=[hq,q]; coeff=[1.0,-alpha]
        if qprev is not None:
            parts.append(qprev);coeff.append(-beta_prev)
        r=mps_linear_combination(parts,coeff)
        # Right compression canonicalizes the direct sum; for zero budget this is exact.
        r,rr=right_compress(r,compression_budget,False)
        beta=math.sqrt(max(0.0,float(np.real(overlap(r,r)))))
        # right_compress stores norm in r.scale, so normalize globally.
        if beta>0:
            qnext=MPS(r.cores,r.bond_sectors,r.scale/beta,r.error_bound/max(beta,1e-300))
        else:qnext=None
        al.append(alpha);reports.append({'k':k,'alpha':alpha,'beta_next':beta,'hb_max_bond':hr['max_bond'],'residual_max_bond':rr['max_bond'],'hb_discard':hr['discarded_l2'],'rc_discard':rr['discarded_l2']})
        if verbose:print('LANCZOS',reports[-1],flush=True)
        if k<m-1:
            be.append(beta)
            if beta<1e-14:break
            qprev,q,beta_prev=q,qnext,beta;qs.append(q)
    return np.array(al,float),np.array(be,float),qs,reports
