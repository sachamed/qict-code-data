#!/usr/bin/env python3
from __future__ import annotations
import os,sys,time,json,multiprocessing as mp
from collections import defaultdict
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k

def worker(chunk):
    out=defaultdict(complex)
    for rep,a in chunk:
        na=len(k.orbit(rep)); loc=defaultdict(complex)
        for pid in range(24):
            for tgt,z in k.wp_action(rep,pid): loc[k.canonical(tgt)] += z
        for crep,z in loc.items(): out[crep] += a*(na/len(k.orbit(crep)))*z
    return dict(out)

def main():
    vac=k.canonical(k.gs.vacuum_state()); v={vac:1+0j}; pre=[]
    for d in range(1,4):
        t=time.perf_counter(); v=k.H_on_orbit_vector(v); pre.append(time.perf_counter()-t)
    items=list(v.items()); nproc=min(4,os.cpu_count() or 1); chunk_size=25
    chunks=[items[i:i+chunk_size] for i in range(0,len(items),chunk_size)]
    agg=defaultdict(complex); t=time.perf_counter()
    ctx=mp.get_context('fork')
    with ctx.Pool(nproc) as pool:
        for part in pool.imap_unordered(worker,chunks,chunksize=1):
            for s,z in part.items(): agg[s]+=z
    dt=time.perf_counter()-t
    v4={s:z for s,z in agg.items() if abs(z)>1e-13}
    m4=v4.get(vac,0j); n2=k.norm2(v4)
    out={'status':'PASS_K0_HB_EXACT_D4' if abs(m4.imag)<1e-10 else 'FAIL',
         'pre_seconds_d1_d3':pre,'workers':nproc,'chunk_size':chunk_size,
         'support_d3':len(v),'support_d4':len(v4),'seconds_d4':dt,'norm2_d4':n2,
         'vacuum_moment_HB4':[m4.real,m4.imag],
         'cache_note':'multiprocessing fork; local recoupling caches retained per worker'}
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
