#!/usr/bin/env python3
from __future__ import annotations
import json, math, itertools, time, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import jv
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; R=ROOT/'results'
L=2
sites=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
links=[(mu,x,y,z) for mu in range(3) for x,y,z in sites]; li={t:i for i,t in enumerate(links)}
plaquettes=[]
for mu in range(3):
 for nu in range(mu+1,3):
  for x,y,z in sites:
   s=[x,y,z];sm=s.copy();sm[mu]=(sm[mu]+1)%L;sn=s.copy();sn[nu]=(sn[nu]+1)%L
   plaquettes.append([(li[(mu,*s)],1),(li[(nu,*sm)],1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)])
inc=[[] for _ in range(24)]
for p,po in enumerate(plaquettes):
 for e,sgn in po: inc[e].append((p,sgn))
order=json.load(open(R/'PATHWIDTH_EXACT_CERTIFICATE.json'))['optimal_order']; pos={e:i for i,e in enumerate(order)}
first=[min(pos[e] for e,_ in po) for po in plaquettes]; last=[max(pos[e] for e,_ in po) for po in plaquettes]

def compress(states,budget):
    if not states: return states,0.0,0,0.0
    keys=list(states.keys()); vals=np.fromiter((abs(states[k])**2 for k in keys),float,count=len(keys))
    n0=len(keys); total=float(vals.sum())
    if budget<=0: return states,0.0,n0,total**.5
    idx=np.argsort(vals) # smallest first
    cs=np.cumsum(vals[idx])
    nd=int(np.searchsorted(cs,budget*budget,side='right'))
    if nd<=0:return states,0.0,n0,total**.5
    # never discard all
    nd=min(nd,n0-1)
    drop=set(int(i) for i in idx[:nd])
    out={k:states[k] for i,k in enumerate(keys) if i not in drop}
    disc=float(vals[list(drop)].sum()**0.5) if drop else 0.0
    return out,disc,len(out),total**.5

def run(K=6,per_step_budget=2e-9):
    x=4*math.pi/63; kval=tuple(range(-K,K+1)); w={k:(1j**k)*jv(k,x) for k in kval}
    states={():1+0j}; opens=[]; profile=[]; disc_sum=0.0; t0=time.time()
    for step,e in enumerate(order):
      oldopens=opens[:]; newp=sorted(p for p in range(24) if first[p]==step); target_open=sorted(p for p in range(24) if first[p]<=step<last[p])
      old_index={p:i for i,p in enumerate(oldopens)};terms=inc[e];pivot=newp[-1] if newp else None;enum_new=[p for p in newp if p!=pivot];sgn={p:s for p,s in terms}
      out=defaultdict(complex)
      for key,a in states.items():
        known=sum(s*key[old_index[p]] for p,s in terms if p in old_index)
        if pivot is None:
          if known:continue
          vals={p:key[old_index[p]] for p in oldopens};out[tuple(vals[p] for p in target_open)]+=a
        else:
          for vv in itertools.product(kval,repeat=len(enum_new)):
            total=known+sum(sgn[p]*v for p,v in zip(enum_new,vv));kp=-total*sgn[pivot]
            if kp < -K or kp > K:continue
            vals={p:key[old_index[p]] for p in oldopens}; vals.update(zip(enum_new,vv)); vals[pivot]=kp; amp=a
            for p in newp: amp*=w[vals[p]]
            out[tuple(vals[p] for p in target_open)]+=amp
      raw=len(out);states=dict(out);opens=target_open
      # compress at every stage; basis is orthonormal in open Fourier labels.
      states,disc,kept,l2=compress(states,per_step_budget)
      disc_sum+=disc
      profile.append({'step':step,'edge':e,'open_plaquettes':len(opens),'raw_states':raw,'kept_states':kept,'discarded_l2':disc,'frontier_l2_before_discard':l2})
      print('K',K,'step',step,'raw',raw,'keep',kept,'disc',disc,flush=True)
    z=states.get((),0j)
    return {'status':'PASS_U1_STATIC_WIDTH14_COMPRESSED_FRONTIER','K':K,'per_step_l2_budget':per_step_budget,
            'sum_local_discarded_l2':disc_sum,'final_scalar':[float(z.real),float(z.imag)],'profile':profile,'seconds':time.time()-t0,
            'scope':'Static U(1) plaquette/Gauss frontier regression only. The coefficient basis of open Fourier labels is orthonormal and each reported discarded_l2 is exact for that boundary tensor. This run does not yet certify propagation of the local truncation error through the repeated-time physical PEPO and is not h_d.'}

def main():
 K=int(sys.argv[1]) if len(sys.argv)>1 else 6; b=float(sys.argv[2]) if len(sys.argv)>2 else 2e-9
 out=run(K,b); fn=R/f'U1_STATIC_FRONTIER_K{K}.json';fn.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'K':K,'final_scalar':out['final_scalar'],'sum_local_discarded_l2':out['sum_local_discarded_l2'],'max_kept':max(x['kept_states'] for x in out['profile']),'seconds':out['seconds']},indent=2))
if __name__=='__main__':main()
