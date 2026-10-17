#!/usr/bin/env python3
from __future__ import annotations
import os,sys,time,json,multiprocessing as mp
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
ROOT=Path(__file__).resolve().parents[1]

def worker(chunk):
    out=defaultdict(complex)
    for rep,a in chunk:
        na=len(k.orbit(rep)); loc=defaultdict(complex)
        for pid in range(24):
            for tgt,z in k.wp_action(rep,pid): loc[k.canonical(tgt)] += z
        for crep,z in loc.items(): out[crep] += a*(na/len(k.orbit(crep)))*z
    return dict(out)

def ecost(st): return sum(n*(n+2) for n in st.links)

def main():
    vac=k.canonical(k.gs.vacuum_state());v={vac:1+0j}
    for _ in range(3): v=k.H_on_orbit_vector(v)
    items=list(v.items()); chunks=[items[i:i+25] for i in range(0,len(items),25)];agg=defaultdict(complex);t=time.perf_counter()
    with mp.get_context('fork').Pool(min(4,os.cpu_count() or 1)) as pool:
      for part in pool.imap_unordered(worker,chunks,chunksize=1):
       for s,z in part.items():agg[s]+=z
    v4={s:z for s,z in agg.items() if abs(z)>1e-13}
    by=defaultdict(float); total=0.0; maxn=0
    for s,a in v4.items():
      w=len(k.orbit(s))*abs(a)**2; c=ecost(s); by[c]+=w;total+=w;maxn=max(maxn,max(s.links))
    # normalized distribution and cumulative tails
    keys=sorted(by); probs={c:by[c]/total for c in keys};tails=[]
    for C in [8,16,24,32,48,64,80,96,128,160,192,256,320,512]:
      tail=sum(p for c,p in probs.items() if c>C)
      tails.append({'electric_cost_cut':C,'discarded_norm':tail**0.5,'discarded_probability':tail})
    q=[];acc=0.0
    for c in keys:
      acc+=probs[c]
      if not q or acc>=q[-1][0]:
       pass
    for target in [0.9,0.99,0.999,0.9999,0.99999,0.999999]:
      acc=0.0;cc=keys[-1]
      for c in keys:
       acc+=probs[c]
       if acc>=target: cc=c;break
      q.append({'cumulative_probability':target,'electric_cost_required':cc})
    out={'status':'PASS_SU2_HB_D4_ELECTRIC_COST_PROFILE','support_d4':len(v4),'norm2_d4':total,'max_twice_j_seen':int(maxn),
         'cost_min':int(keys[0]),'cost_max':int(keys[-1]),'tails':tails,'quantiles':q,
         'profile_seconds_d4':time.perf_counter()-t,
         'scientific_scope':'Exact weighted-l2 cost profile of H_B^4|vac> in the translation-orbit spin-network basis. This diagnoses whether a Casimir-block compression could be efficient; it is not itself a frontier or physical h_d calculation.'}
    (ROOT/'results'/'SU2_HB_D4_ELECTRIC_COST_PROFILE.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
