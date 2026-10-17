from __future__ import annotations
import numpy as np,itertools,json,hashlib
from pathlib import Path
L=2; coords=list(itertools.product(range(L),repeat=3));links=[(mu,*x) for mu in range(3) for x in coords];li={t:i for i,t in enumerate(links)}
sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]],complex);sz=np.array([[1,0],[0,-1]],complex);I=np.eye(2,dtype=complex);PROJ=[((I+s)/2,(I-s)/2) for s in [sx,sy,sz]]
def move(x,mu,s):a=list(x);a[mu]=(a[mu]+s)%L;return tuple(a)
def step(st,mu,tol=1e-15):
 o={};Pp,Pm=PROJ[mu]
 for (x,J),M in st.items():
  y=move(x,mu,1);jj=list(J);jj[li[(mu,*x)]]+=1;k=(y,tuple(jj));A=Pm@M
  if np.linalg.norm(A)>tol:o[k]=o.get(k,0)+A
  y=move(x,mu,-1);jj=list(J);jj[li[(mu,*y)]]-=1;k=(y,tuple(jj));A=Pp@M
  if np.linalg.norm(A)>tol:o[k]=o.get(k,0)+A
 return {k:v for k,v in o.items() if np.linalg.norm(v)>tol}
def macro(st):
 for mu in (2,1,0):st=step(st,mu)
 return st
def div(J):
 d={x:0 for x in coords}
 for mu,*xx in links:
  x=tuple(xx);y=move(x,mu,1);v=J[li[(mu,*x)]];d[x]-=v;d[y]+=v
 return tuple(d[x] for x in coords)
def wind(J):return tuple(int(sum(J[8*m:8*(m+1)])//2) for m in range(3))
st={((0,0,0),(0,)*24):np.eye(2,dtype=complex)}
for d in range(9):st=macro(st)
assert len(st)==20416
items=list(st.items()); endpoint=(1,1,1);assert all(k[0]==endpoint for k,v in items)
Jref=np.zeros(24,dtype=int)
# canonical open reference: +z, then +y, then +x from the origin, matching one C8 macro ordering
x=(0,0,0)
for mu in (2,1,0):
 Jref[li[(mu,*x)]]+=1
 x=move(x,mu,1)
assert x==endpoint
dvref=div(Jref)
hist={};zero=[];amp=0j
for (x,Jt),M in items:
 J=np.array(Jt,int);D=J-Jref
 assert all(v==0 for v in div(D))
 w=wind(D);hist[w]=hist.get(w,0)+1
 if w==(0,0,0):
  a=np.trace(M)/2;zero.append((tuple(Jt),a));amp+=a
# Also classify zero-relative currents by the boundary current D using exact tuples; raw unique already.
obj={'status':'PASS_D9_RELATIVE_HOMOLOGY_SELECTION','route':9,'endpoint':endpoint,'total_currents':len(items),
     'reference_current':Jref.tolist(),'reference_divergence':[int(x) for x in dvref],'relative_harmonic_sector_count':len(hist),
     'zero_relative_winding_count':len(zero),'zero_relative_winding_fraction':len(zero)/len(items),
     'zero_relative_spin_trace_amplitude':[float(amp.real),float(amp.imag)],
     'harmonic_histogram':{str(k):v for k,v in sorted(hist.items())},
     'theorem_use':'For a neutral vacuum supported in one harmonic electric sector, only route currents whose difference from a fixed external-source current is a plaquette boundary can contribute to the same-sector return.'}
txt=json.dumps(obj,indent=2,sort_keys=True)+'\n';Path('/mnt/data/task2_continue/D9_RELATIVE_HOMOLOGY_SELECTION.json').write_text(txt)
print(json.dumps({k:obj[k] for k in ('status','total_currents','relative_harmonic_sector_count','zero_relative_winding_count','zero_relative_winding_fraction','zero_relative_spin_trace_amplitude')},indent=2));print('sha256',hashlib.sha256(txt.encode()).hexdigest())
