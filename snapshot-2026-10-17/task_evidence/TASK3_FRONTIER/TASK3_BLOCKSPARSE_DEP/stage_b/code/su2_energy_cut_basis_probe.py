#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,time,collections
from pathlib import Path
sys.path.insert(0,os.path.dirname(__file__))
import su2_k0_orbit_krylov as k
import su2_global_spin_network as gs
ROOT=Path(__file__).resolve().parents[1]

def ecost(st): return sum(n*(n+2) for n in st.links)

def neighbors(rep):
 out={}
 for pid in range(24):
  for tgt,z in k.wp_action(rep,pid):
   c=k.canonical(tgt); out[c]=out.get(c,0j)+z
 return out

def run(Emax,max_states=300000):
 vac=k.canonical(gs.vacuum_state()); seen={vac}; q=collections.deque([vac]); hist=collections.Counter({0:1}); transitions=0; boundary=0;t0=time.time();
 while q:
  s=q.popleft();
  for t,z in neighbors(s).items():
   if abs(z)<1e-14:continue
   transitions+=1;e=ecost(t)
   if e<=Emax:
    if t not in seen:
     seen.add(t);q.append(t);hist[e]+=1
     if len(seen)>=max_states:
      return {'status':'STOP_MAX_STATES','Emax':Emax,'orbit_states':len(seen),'queue_remaining':len(q),'transitions_seen':transitions,'electric_cost_histogram':dict(sorted(hist.items())),'seconds':time.time()-t0}
   else: boundary+=1
 return {'status':'PASS_SU2_ENERGY_CUT_BASIS_CLOSED_ENUMERATION','Emax':Emax,'orbit_states':len(seen),'transitions_seen':transitions,'boundary_transition_count':boundary,'electric_cost_histogram':dict(sorted(hist.items())),'seconds':time.time()-t0}
if __name__=='__main__':
 E=int(sys.argv[1]);M=int(sys.argv[2]) if len(sys.argv)>2 else 300000
 out=run(E,M);(ROOT/'results'/f'SU2_ENERGY_CUT_BASIS_E{E}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
