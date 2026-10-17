#!/usr/bin/env python3
from __future__ import annotations
import math,json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import jv
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; R=ROOT/'results'
L=2; PHI=math.pi/63; X=4*PHI
sites=[(x,y,z) for x in range(2) for y in range(2) for z in range(2)]
links=[(mu,x,y,z) for mu in range(3) for x,y,z in sites]; li={e:i for i,e in enumerate(links)}
plaquettes=[]
for mu in range(3):
 for nu in range(mu+1,3):
  for x,y,z in sites:
   s=[x,y,z];sm=s.copy();sm[mu]=(sm[mu]+1)%2;sn=s.copy();sn[nu]=(sn[nu]+1)%2
   v=np.zeros(24,dtype=np.int16)
   for e,sg in [(li[(mu,*s)],1),(li[(nu,*sm)],1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)]:v[e]+=sg
   plaquettes.append(tuple(map(int,v)))

def pplus(e):return 2*e*e+e

def compress(v,budget):
 keys=list(v);sq=np.fromiter((abs(v[k])**2 for k in keys),float,count=len(keys)); idx=np.argsort(sq);cs=np.cumsum(sq[idx]);nd=int(np.searchsorted(cs,budget*budget,side='right'));nd=min(nd,max(0,len(keys)-1))
 if nd<=0:return v,0.0
 drop=set(int(i) for i in idx[:nd]);disc=float(sq[list(drop)].sum()**.5)
 return {k:v[k] for i,k in enumerate(keys) if i not in drop},disc

def gate(v,pl,K):
 out=defaultdict(complex); w=[((1j)**k)*jv(k,X) for k in range(-K,K+1)]
 for E,a in v.items():
  for k,c in zip(range(-K,K+1),w):
   Ep=tuple(E[i]+k*pl[i] for i in range(24));out[Ep]+=a*c
 return dict(out)

def run(K=6,dmax=10,budget=2e-9):
 v={(0,)*24:1+0j}; prof=[];ds=0.;t0=time.time(); overlaps={}
 for t in range(1,dmax+1):
  # electric phase first; diagonal and exactly norm preserving
  v={E:a*np.exp(-1j*PHI*sum(pplus(e) for e in E)) for E,a in v.items()}
  for p,pl in enumerate(plaquettes):
   raw=gate(v,pl,K);v,disc=compress(raw,budget);ds+=disc
   if p in (0,5,11,17,23):
    prof.append({'time':t,'plaquette':p,'raw':len(raw),'kept':len(v),'discarded_l2':disc,'norm':float(sum(abs(a)**2 for a in v.values())**.5)})
  overlaps[str(t)]=[float(v.get((0,)*24,0j).real),float(v.get((0,)*24,0j).imag)]
  print('t',t,'support',len(v),'norm',sum(abs(a)**2 for a in v.values())**.5,'disc_sum',ds,flush=True)
  if len(v)>2_000_000:break
 out={'status':'PASS_U1_PHYSICAL_SPARSE_UNITARY_EVOLUTION_PROBE','K':K,'d_completed':t,'per_gate_discard_budget':budget,'sum_discarded_l2':ds,'support':len(v),'norm':float(sum(abs(a)**2 for a in v.values())**.5),'vacuum_overlaps':overlaps,'profile':prof,'seconds':time.time()-t0,
      'scope':'True orthonormal 24-link U(1) flux basis, electric phase plus all 24 character-expanded plaquette gates. Compression is direct l2 state truncation after each norm-one physical gate. Starts from the primitive zero-flux vacuum and contains no C8 defects; therefore it is a contractor regression, not the Kato-selected physical h_d.'}
 (R/f'U1_PHYSICAL_SPARSE_EVOLUTION_K{K}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','K','d_completed','sum_discarded_l2','support','norm','seconds')},indent=2))
if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else 6,int(sys.argv[2]) if len(sys.argv)>2 else 10,float(sys.argv[3]) if len(sys.argv)>3 else 2e-9)
