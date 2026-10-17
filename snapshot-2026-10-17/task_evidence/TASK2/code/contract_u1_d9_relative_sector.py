from __future__ import annotations
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/task2_continue')
import contract_u1_trivial_orbits as u1
# Reuse route enumerator definitions without executing file output logic: reimplement concise
L=2;coords=u1.sites;links=u1.links;li=u1.li
sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]],complex);sz=np.array([[1,0],[0,-1]],complex);I=np.eye(2,dtype=complex);PROJ=[((I+s)/2,(I-s)/2) for s in (sx,sy,sz)]
def move(x,mu,s):a=list(x);a[mu]=(a[mu]+s)%2;return tuple(a)
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
 for mu,x,y,z in links:
  a=(x,y,z);b=move(a,mu,1);v=J[li[(mu,*a)]];d[a]-=v;d[b]+=v
 return tuple(d[x] for x in coords)
def wind(J):return tuple(int(sum(J[8*m:8*(m+1)])//2) for m in range(3))
# d=9 expansion
st={((0,0,0),(0,)*24):np.eye(2,dtype=complex)}
for _ in range(9):st=macro(st)
endpoint=(1,1,1)
Jref=np.zeros(24,dtype=int);x=(0,0,0)
for mu in (2,1,0):Jref[li[(mu,*x)]]+=1;x=move(x,mu,1)
assert x==endpoint
sel=[]
for (x,Jt),M in st.items():
 D=np.asarray(Jt,dtype=int)-Jref
 if wind(D)==(0,0,0):
  assert all(v==0 for v in div(D))
  sel.append((D,complex(np.trace(M)/2)))
assert len(sel)==102
# exact gauge factors for closed relative current D
K=6; bv=u1.base.one(K); zv=complex(bv['Z_real'],bv['Z_imag'])
t0=time.time();total=0j;rows=[]
for i,(D,amp) in enumerate(sel):
 z,k0,offs=u1.evalJ(D.tolist(),K,-3); ratio=z/zv; contrib=amp*ratio; total+=contrib
 rows.append({'i':i,'relative_current':D.tolist(),'spin_trace_amp':[amp.real,amp.imag],'gauge_ratio':[ratio.real,ratio.imag],'contribution':[contrib.real,contrib.imag]})
 if (i+1)%10==0: print(i+1,'/',len(sel),'partial',total,flush=True)
out={'status':'PASS_U1_D9_RELATIVE_SOURCE_SECTOR_EXACT','external_particle_data_used':False,'route':9,'K':K,'q':-3,
     'source_reference':'minimal +z,+y,+x path from (0,0,0) to (1,1,1)','selected_zero_relative_winding_currents':len(sel),
     'Z_vac':[zv.real,zv.imag],'route_amplitude_relative_source':[total.real,total.imag],'abs_route_amplitude':abs(total),
     'seconds':time.time()-t0,'rows':rows,
     'scope':'Exact static primitive-Haar U(1) character contraction of the gauge-invariant relative route/source loops. It is a spatial regression, not the full Kato temporal PEPO amplitude.'}
txt=json.dumps(out,indent=2,sort_keys=True)+'\n';Path('/mnt/data/task2_continue/U1_D9_RELATIVE_SOURCE_EXACT.json').write_text(txt)
print(json.dumps({k:out[k] for k in ('status','selected_zero_relative_winding_currents','route_amplitude_relative_source','abs_route_amplitude','seconds')},indent=2));print('sha256',hashlib.sha256(txt.encode()).hexdigest())
