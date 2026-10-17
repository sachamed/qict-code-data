#!/usr/bin/env python3
from __future__ import annotations
import os,sys,time,json,multiprocessing as mp
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
ROOT=Path(__file__).resolve().parent.parent

def e_cost(st): return sum(n*(n+2) for n in st.links)
def inner(a,b): return sum(len(k.orbit(s))*complex(a.get(s,0)).conjugate()*z for s,z in b.items())
def norm(v): return float(max(0.0,inner(v,v).real))**0.5
def scale(v,a): return {s:a*z for s,z in v.items() if abs(a*z)>1e-16}
def axpy(y,a,x):
    z=dict(y)
    for s,v in x.items(): z[s]=z.get(s,0j)+a*v
    return {s:v for s,v in z.items() if abs(v)>1e-14}  # corrected, currently unused

def combine(*terms):
    z=defaultdict(complex)
    for a,v in terms:
        for s,x in v.items(): z[s]+=a*x
    return {s:x for s,x in z.items() if abs(x)>1e-14}

def h_worker(chunk):
    out=defaultdict(complex)
    for rep,a in chunk:
        na=len(k.orbit(rep)); loc=defaultdict(complex)
        for pid in range(24):
            for tgt,z in k.wp_action(rep,pid): loc[k.canonical(tgt)] += z
        for crep,z in loc.items(): out[crep] += a*(na/len(k.orbit(crep)))*z
    return dict(out)

def H(v):
    if len(v)<400: return k.H_on_orbit_vector(v)
    items=list(v.items()); chunks=[items[i:i+20] for i in range(0,len(items),20)]
    agg=defaultdict(complex)
    ctx=mp.get_context('fork')
    with ctx.Pool(min(4,os.cpu_count() or 1)) as pool:
        for part in pool.imap_unordered(h_worker,chunks,chunksize=1):
            for s,z in part.items(): agg[s]+=z
    return {s:z for s,z in agg.items() if abs(z)>1e-14}

def K(v):
    hv=H(v); out=defaultdict(complex)
    for s,a in v.items(): out[s]+=e_cost(s)*a
    for s,a in hv.items(): out[s]+=-16*a
    return {s:a for s,a in out.items() if abs(a)>1e-14}

def tail_profile(v):
    # Norm contribution of an orbit coefficient is |O| |a|^2.
    vals=sorted((len(k.orbit(s))*abs(a)**2 for s,a in v.items()),reverse=True)
    tot=sum(vals); out={}
    for Kkeep in (100,300,1000,3000,10000,30000):
        if Kkeep>=len(vals): tail=0.0
        else: tail=sum(vals[Kkeep:])
        out[str(Kkeep)]={'discarded_norm':float(tail**0.5),'discarded_prob_fraction':float(tail/tot if tot else 0)}
    for eps in (1e-4,1e-6,1e-8,1e-10):
        acc=0.0; n=len(vals)
        for i,x in enumerate(reversed(vals)):
            if acc+x > eps*eps: n=len(vals)-i; break
            acc+=x
        out[f'budget_{eps:g}']={'minimum_kept_by_weight_bound':n,'actual_discarded_norm':float(acc**0.5)}
    return out

def main(iters=4):
    vac=k.canonical(k.gs.vacuum_state()); q={vac:1+0j}; qm1={}; beta_prev=0.0
    rows=[]; al=[]; be=[]
    for j in range(iters):
        t=time.perf_counter(); w=K(q); dt=time.perf_counter()-t
        alpha=inner(q,w).real; al.append(float(alpha))
        w=combine((1,w),(-alpha,q),(-beta_prev,qm1))
        # one reorthogonalization against current and previous vectors
        cq=inner(q,w)
        if abs(cq)>1e-12: w=combine((1,w),(-cq,q))
        if qm1:
            cp=inner(qm1,w)
            if abs(cp)>1e-12: w=combine((1,w),(-cp,qm1))
        beta=norm(w); be.append(float(beta))
        row={'iteration':j,'q_support':len(q),'Kq_support':len(w),'alpha':float(alpha),'beta_next':float(beta),'matvec_seconds':dt,'q_tail_profile':tail_profile(q)}
        rows.append(row)
        if beta<1e-13: break
        qm1,q=q,{s:a/beta for s,a in w.items()}; beta_prev=beta
    # final q tail
    final={'support':len(q),'tail_profile':tail_profile(q)}
    out={'status':'PASS_SU2_KPRIM_LANCZOS_TAIL_PROFILE','iterations_completed':len(rows),'alphas':al,'betas':be,'rows':rows,'final_q':final,
         'note':'No compression is applied in this profiling run. Tail norms are exact weighted-l2 diagnostics in the translation-orbit basis and may be used to choose a later certified sparse compression budget; they are not yet a PEPO frontier error certificate.'}
    (ROOT/'results'/'SU2_KPRIM_LANCZOS_TAIL_PROFILE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 4)
