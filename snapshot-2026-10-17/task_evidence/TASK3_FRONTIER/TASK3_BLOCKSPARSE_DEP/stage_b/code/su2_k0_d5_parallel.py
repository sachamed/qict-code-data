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

def parallel_apply(items,nproc=5,chunk_size=50):
    chunks=[items[i:i+chunk_size] for i in range(0,len(items),chunk_size)]
    agg=defaultdict(complex); ctx=mp.get_context('fork'); t=time.perf_counter();done=0
    with ctx.Pool(nproc) as pool:
        for part in pool.imap_unordered(worker,chunks,chunksize=1):
            for s,z in part.items(): agg[s]+=z
            done+=1
    return {s:z for s,z in agg.items() if abs(z)>1e-12},time.perf_counter()-t,len(chunks)

def main():
    vac=k.canonical(k.gs.vacuum_state());v={vac:1+0j};pre=[]
    for d in range(1,4):
        t=time.perf_counter();v=k.H_on_orbit_vector(v);pre.append(time.perf_counter()-t)
    v4,t4,c4=parallel_apply(list(v.items()),min(5,os.cpu_count() or 1),50)
    # consistency with prior exact shell
    m4=v4.get(vac,0j); n4=k.norm2(v4)
    if abs(m4-1713)>2e-8: raise RuntimeError(('m4 mismatch',m4))
    v5,t5,c5=parallel_apply(list(v4.items()),min(5,os.cpu_count() or 1),40)
    m5=v5.get(vac,0j); n5=k.norm2(v5)
    out={'status':'PASS_K0_HB_EXACT_D5' if abs(m5)<2e-8 else 'FAIL',
      'pre_seconds_d1_d3':pre,'support_d3':len(v),'support_d4':len(v4),'support_d5':len(v5),
      'seconds_d4':t4,'seconds_d5':t5,'chunks_d4':c4,'chunks_d5':c5,
      'vacuum_moment_HB4':[m4.real,m4.imag],'vacuum_moment_HB5':[m5.real,m5.imag],
      'norm2_d4':n4,'norm2_d5':n5,
      'm10_from_hermiticity_norm_H5':n5,
      'workers':min(5,os.cpu_count() or 1),
      'representation':'exact translation-orbit point-split spin-network basis; no amplitude truncation except numerical zero filter 1e-12'}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
