#!/usr/bin/env python3
import os,sys,time,json,multiprocessing as mp
from collections import defaultdict
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
from su2_k0_d4_parallel import worker

def main():
 vac=k.canonical(k.gs.vacuum_state());v={vac:1+0j}
 for _ in range(3):v=k.H_on_orbit_vector(v)
 items=list(v.items());ctx=mp.get_context('fork');agg=defaultdict(complex)
 with ctx.Pool(4) as pool:
  for part in pool.imap_unordered(worker,[items[i:i+25] for i in range(0,len(items),25)]):
   for s,z in part.items():agg[s]+=z
 v4={s:z for s,z in agg.items() if abs(z)>1e-13}
 reps=list(v4.items())
 rows=[]
 for n in (10,25,50,100):
  t=time.perf_counter();raw=0;uniq=set()
  for rep,a in reps[:n]:
   for pid in range(24):
    aa=k.wp_action(rep,pid);raw+=len(aa)
    for tgt,z in aa:uniq.add(k.canonical(tgt))
  rows.append({'sample':n,'seconds':time.perf_counter()-t,'raw_transitions':raw,'unique_targets':len(uniq)})
 print(json.dumps({'status':'PASS_D5_COST_SAMPLE','support_d4':len(v4),'rows':rows},indent=2))
if __name__=='__main__':main()
