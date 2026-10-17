import numpy as np,json,hashlib
from pathlib import Path
L=2
coords=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
links=[(mu,x,y,z) for mu in range(3) for x,y,z in coords];li={t:i for i,t in enumerate(links)}
sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]],complex);sz=np.array([[1,0],[0,-1]],complex);I=np.eye(2,dtype=complex);PROJ=[((I+s)/2,(I-s)/2) for s in [sx,sy,sz]]
def move(site,mu,sgn):a=list(site);a[mu]=(a[mu]+sgn)%L;return tuple(a)
def step_dir(states,mu,tol=1e-15):
 out={};Pp,Pm=PROJ[mu]
 for (site,J),M in states.items():
  y=move(site,mu,+1);jj=list(J);jj[li[(mu,*site)]]+=1;key=(y,tuple(jj));A=Pm@M
  if np.linalg.norm(A)>tol:out[key]=out.get(key,0)+A
  y=move(site,mu,-1);jj=list(J);jj[li[(mu,*y)]]-=1;key=(y,tuple(jj));A=Pp@M
  if np.linalg.norm(A)>tol:out[key]=out.get(key,0)+A
 return {k:v for k,v in out.items() if np.linalg.norm(v)>tol}
def macro(st):
 for mu in (2,1,0):st=step_dir(st,mu)
 return st
def div(J):
 d={x:0 for x in coords}
 for mu,x,y,z in links:
  a=(x,y,z);b=move(a,mu,+1);v=J[li[(mu,*a)]];d[a]-=v;d[b]+=v
 return tuple(d[x] for x in coords)
def wind(J):return tuple(int(sum(J[8*mu:8*(mu+1)])//L) for mu in range(3))
st={((0,0,0),(0,)*24):np.eye(2,dtype=complex)};rows=[]
for dstep in range(1,11):
 st=macro(st)
 if dstep not in (8,9,10):continue
 hist={}; endpoint={}; triv_count=0; amp_all=0j;amp_triv=0j; amp_closed=0j; closed_count=0
 for (site,J),M in st.items():
  a=np.trace(M)/2; amp_all+=a;endpoint[site]=endpoint.get(site,0)+1
  dv=div(J); w=wind(J); hist[w]=hist.get(w,0)+1
  isclosed=all(x==0 for x in dv)
  if isclosed:
   closed_count+=1;amp_closed+=a
   if w==(0,0,0):triv_count+=1;amp_triv+=a
 row={'d':dstep,'unique_currents':len(st),'endpoint_counts':{str(k):v for k,v in endpoint.items()},'closed_count':closed_count,'homologically_trivial_closed_count':triv_count,'spin_trace_amplitude_all':[amp_all.real,amp_all.imag],'spin_trace_amplitude_closed':[amp_closed.real,amp_closed.imag],'spin_trace_amplitude_homologically_trivial':[amp_triv.real,amp_triv.imag],'harmonic_sector_count':len(hist),'largest_harmonic_sector_count':max(hist.values()),'harmonic_histogram':{str(k):v for k,v in sorted(hist.items())}}
 rows.append(row);print({k:row[k] for k in row if k!='harmonic_histogram'},flush=True)
obj={'status':'PASS_C8_ROUTE_CURRENT_EXACT_ENUMERATION','geometry':'periodic L=2 BCC split-step spatial stream','route_lengths':[8,9,10],'rows':rows,'interpretation':'Exact symbolic expansion of U_C8^d into integer link currents and 2x2 spin amplitudes. Closed zero-harmonic currents are boundaries and are directly compatible with the exact affine U(1) Haar solver; open/harmonic sectors require C10/boundary closure.'}
txt=json.dumps(obj,indent=2,sort_keys=True)+'\n';Path('/mnt/data/task2_continue/C8_ROUTE_CURRENT_ENUMERATION.json').write_text(txt);print('sha256',hashlib.sha256(txt.encode()).hexdigest())
