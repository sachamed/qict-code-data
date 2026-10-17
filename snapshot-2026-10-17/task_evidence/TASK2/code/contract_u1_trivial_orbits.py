from __future__ import annotations
import sys,json,time,math,hashlib
from pathlib import Path
import numpy as np
from scipy.special import jv
T1=Path('/mnt/data/task1_reextract/QICT_D21_TASK1_MICROSCOPIC_OPERATOR_CLOSURE_PASS_CLEAN/code/microscopic_closure');sys.path.insert(0,str(T1))
import compute_u1_l2_exact_character as base
PHI=math.pi/63;X=4*PHI;NPLAQ=24
# geometry copied consistently
L=2;sites=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)];links=[(mu,x,y,z) for mu in range(3) for x,y,z in sites];li={t:i for i,t in enumerate(links)}
pmeta=[];plaquettes=[]
for mu in range(3):
 for nu in range(mu+1,3):
  for x,y,z in sites:
   s=[x,y,z];sm=s.copy();sm[mu]^=1;sn=s.copy();sn[nu]^=1
   plaquettes.append([(li[(mu,*s)],1),(li[(nu,*sm)],1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)]);pmeta.append((mu,nu,x,y,z))
B=np.zeros((24,24),dtype=np.int64)
for p,po in enumerate(plaquettes):
 for l,sg in po:B[l,p]+=sg
PC=[0,1,2,3,4,5,8,9,10,11,12,16,17,18]; PR=[0,1,2,4,5,6,8,9,10,12,14,16,18,20]
A=B[np.ix_(PR,PC)]
Ainv=np.rint(np.linalg.inv(A)).astype(np.int64)
assert np.array_equal(A@Ainv,np.eye(14,dtype=np.int64))
PID={t:i for i,t in enumerate(pmeta)}
def particular(J,q=-3):
 y=-q*np.asarray(J,dtype=np.int64); x=Ainv@y[PR];k=np.zeros(24,dtype=np.int64);k[PC]=x
 if not np.array_equal(B@k,y):raise ValueError('current not in boundary image')
 return k
def offsets(k0):
 out={};muv=[(0,1),(0,2),(1,2)]
 for r in range(3):
  mu,nu=next(pair for pair in muv if 3-pair[0]-pair[1]==r);oth=[a for a in range(3) if a!=r];pairs=[(0,0),(1,0),(0,1),(1,1)];aa=[];bb=[]
  for u,v in pairs:
   s=[0,0,0];s[r]=0;s[oth[0]]=u;s[oth[1]]=v;t0=(mu,nu,*s);s[r]=1;t1=(mu,nu,*s);aa.append(int(k0[PID[t0]]));bb.append(int(k0[PID[t1]]))
  eps=1 if (mu,nu,r) in ((0,1,2),(1,2,0),(2,0,1)) else -1;out[r]=(eps,tuple(aa),tuple(bb))
 return out
_cache={}
def df(K,eps,aoff,boff):
 key=(K,eps,aoff,boff)
 if key in _cache:return _cache[key]
 D=2*K+1;wt=np.asarray([(1j**k)*jv(k,X) for k in range(-K,K+1)],complex);F=np.zeros(D**4,complex)
 def W(k):return wt[k+K] if -K<=k<=K else 0j
 idx=0
 for d0 in range(-K,K+1):
  for d1 in range(-K,K+1):
   for d2 in range(-K,K+1):
    for d3 in range(-K,K+1):
     ds=(d0,d1,d2,d3);bv=1+0j
     for i,d in enumerate(ds):
      bv*=W(boff[i]-eps*d)
      if bv==0:break
     sh=0j
     if bv!=0:
      lo=-10**9;hi=10**9
      for i,d in enumerate(ds):
       a=aoff[i]
       if eps==1:l=-K-a-d;h=K-a-d
       else:l=a-d-K;h=a-d+K
       lo=max(lo,l);hi=min(hi,h)
      for h in range(lo,hi+1):
       pp=1+0j
       for i,d in enumerate(ds):
        pp*=W(aoff[i]+eps*(h+d))
        if pp==0:break
       sh+=pp
     F[idx]=bv*sh;idx+=1
 _cache[key]=F;return F
def evalJ(J,K=6,q=-3):
 k=particular(J,q);o=offsets(k);Fs=[df(K,*o[r]) for r in range(3)];raw=base.contract_cube(K,Fs[0],Fs[1],Fs[2]);z=np.exp(-1j*X*NPLAQ)*raw;return z,k,o
obj=json.load(open('/mnt/data/task2_continue/C8_TRIVIAL_CURRENT_ORBITS.json')); K=6;bv=base.one(K);zv=complex(bv['Z_real'],bv['Z_imag'])
out={'status':'PASS_U1_EXACT_TRIVIAL_ROUTE_ORBITS','K':K,'q':-3,'Z_vac':[zv.real,zv.imag],'routes':{}}
for ds in ['8','10']:
 total=0j;rows=[];t0=time.time()
 for i,r in enumerate(obj['routes'][ds]['orbits']):
  z,k,o=evalJ(r['current'],K,-3); amp=complex(*r['spin_trace_amplitude_sum']); contrib=amp*z/zv;total+=contrib
  rows.append({'orbit':i,'multiplicity':r['multiplicity_in_enumeration'],'spin_amp_sum':r['spin_trace_amplitude_sum'],'k0':k.tolist(),'normalized_gauge_factor':[float((z/zv).real),float((z/zv).imag)],'contribution':[float(contrib.real),float(contrib.imag)]})
  print('d',ds,'orbit',i,'|ratio|',abs(z/zv),'contrib',contrib,flush=True)
 out['routes'][ds]={'orbit_count':len(rows),'partial_trivial_route_amplitude':[total.real,total.imag],'abs_partial':abs(total),'free_trivial_spin_amplitude':obj['routes'][ds]['spin_trace_amp_sum'],'seconds':time.time()-t0,'orbits':rows}
txt=json.dumps(out,indent=2,sort_keys=True)+'\n';Path('/mnt/data/task2_continue/U1_EXACT_TRIVIAL_ROUTE_ORBITS.json').write_text(txt);print(json.dumps({d:{k:v for k,v in out['routes'][d].items() if k!='orbits'} for d in out['routes']},indent=2));print('sha256',hashlib.sha256(txt.encode()).hexdigest())
