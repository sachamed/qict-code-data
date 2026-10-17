from __future__ import annotations
import math, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path('/mnt/data/QICT_D21_CLOSURE_EXECUTED_2026-09-16')
sys.path.insert(0,str(ROOT/'code'/'final_closure'))
from qict_route_first import electron_route_coefficients, I3, X, Z

PHI=math.pi/63
A1=4.0; A2=32.0; L=2; NS=8
COEFF=electron_route_coefficients(); WF=[I3,X,Z]; D=[10,9,8]
# link indexing same as gauge contraction
links=[(mu,x,y,z) for mu in range(3) for x in range(L) for y in range(L) for z in range(L)]
li={t:i for i,t in enumerate(links)}
TREE=[10,17,5,6,19,7,14]
NUM=[i for i in range(24) if i not in set(TREE)]
# plaquettes
plaquettes=[]
for mu in range(3):
  for nu in range(mu+1,3):
    for x in range(L):
      for y in range(L):
        for z in range(L):
          s=[x,y,z]; sm=s.copy(); sm[mu]=(sm[mu]+1)%L; sn=s.copy(); sn[nu]=(sn[nu]+1)%L
          plaquettes.append([(li[(mu,*s)],1),(li[(nu,*sm)],1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)])

def idx(x,y,z): return (x*L+y)*L+z
coords=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
plus=np.zeros((3,NS),int); minus=np.zeros((3,NS),int)
for mu in range(3):
  for i,(x,y,z) in enumerate(coords):
    a=[x,y,z]; b=a.copy(); b[mu]=(b[mu]+1)%L; plus[mu,i]=idx(*b)
    b=a.copy(); b[mu]=(b[mu]-1)%L; minus[mu,i]=idx(*b)

sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]],complex); sz=np.array([[1,0],[0,-1]],complex)
SIG=[sx,sy,sz]; I2=np.eye(2,dtype=complex); PROJ=[((I2+s)/2,(I2-s)/2) for s in SIG]

# quaternion q=(w,x,y,z)
def su2_from_u(u):
  u1,u2,u3=u[...,0],u[...,1],u[...,2]
  x=np.sqrt(1-u1)*np.sin(2*np.pi*u2); y=np.sqrt(1-u1)*np.cos(2*np.pi*u2)
  z=np.sqrt(u1)*np.sin(2*np.pi*u3); w=np.sqrt(u1)*np.cos(2*np.pi*u3)
  return np.stack([w,x,y,z],axis=-1)
def qconj(q):
  r=q.copy(); r[...,1:]*=-1; return r
def qmul(a,b):
  w1=a[...,0];v1=a[...,1:];w2=b[...,0];v2=b[...,1:]
  w=w1*w2-np.sum(v1*v2,axis=-1)
  v=w1[...,None]*v2+w2[...,None]*v1+np.cross(v1,v2)
  return np.concatenate([w[...,None],v],axis=-1)
def qmat(q):
  w,x,y,z=np.moveaxis(q,-1,0)
  M=np.empty(q.shape[:-1]+(2,2),complex)
  M[...,0,0]=w+1j*z; M[...,0,1]=y+1j*x
  M[...,1,0]=-y+1j*x; M[...,1,1]=w-1j*z
  return M

def gauge_phase(th,Q):
  # th (B,24), Q (B,24,4)
  B=th.shape[0]; sumc=np.zeros(B); sumw=np.zeros(B)
  for po in plaquettes:
    ang=np.zeros(B); prod=np.zeros((B,4));prod[:,0]=1
    for j,sgn in po:
      ang += sgn*th[:,j]
      qq=Q[:,j] if sgn>0 else qconj(Q[:,j])
      prod=qmul(prod,qq)
    sumc += np.cos(ang); sumw += prod[:,0]
  K1=A1*(24-sumc); K2=A2*(24-sumw)
  return np.exp(-1j*PHI*(K1+K2)), K1, K2

def apply_dir_L(psi,mu,u1,Q2):
  # psi B,site,spin,weak,src ; u1 B,24 complex ; Q2 B,24,2,2
  Pp,Pm=PROJ[mu]
  vp=np.einsum('ab,nibwc->niawc',Pp,psi,optimize=True)
  vm=np.einsum('ab,nibwc->niawc',Pm,psi,optimize=True)
  out=np.zeros_like(psi)
  for y in range(NS):
    # Pminus uses Tplus: source x=y-e, link at x
    x=minus[mu,y]; l=li[(mu,*coords[x])]
    R=(u1[:,l]**(-3))[:,None,None]*Q2[:,l]
    out[:,y]+=np.einsum('nuv,nisvc->nisuc',R,vm[:,x:x+1],optimize=True)[:,0]
    # Pplus uses Tminus: source x=y+e, link at y dagger
    x=plus[mu,y]; l=li[(mu,*coords[y])]
    R=(np.conj(u1[:,l])**(-3))[:,None,None]*np.swapaxes(Q2[:,l].conj(),-1,-2)
    out[:,y]+=np.einsum('nuv,nisvc->nisuc',R,vp[:,x:x+1],optimize=True)[:,0]
  return out

def apply_dir_R(psi,mu,u1):
  # B,site,spin,src
  Pp,Pm=PROJ[mu]
  vp=np.einsum('ab,nibc->niac',Pp,psi,optimize=True)
  vm=np.einsum('ab,nibc->niac',Pm,psi,optimize=True)
  out=np.zeros_like(psi)
  for y in range(NS):
    x=minus[mu,y]; l=li[(mu,*coords[x])]
    amp=u1[:,l]**(-6)
    out[:,y]+=amp[:,None,None]*vm[:,x]
    x=plus[mu,y]; l=li[(mu,*coords[y])]
    amp=np.conj(u1[:,l])**(-6)
    out[:,y]+=amp[:,None,None]*vp[:,x]
  return out

def step(Lpsi,Rpsi,u1,Q2):
  # rightmost z then y then x so resulting operator Sx Sy Sz
  for mu in (2,1,0):
    Lpsi=apply_dir_L(Lpsi,mu,u1,Q2); Rpsi=apply_dir_R(Rpsi,mu,u1)
  # exact Clebsch involution h0=(0,1): swap lower weak with R
  Ln=Lpsi.copy(); Rn=Lpsi[:,:,:,1,:].copy(); Ln[:,:,:,1,:]=Rpsi
  return Ln,Rn

def init_sources(B):
  # four source columns: (L,s=0),(L,s=1),(R,s=0),(R,s=1)
  Lp=np.zeros((B,NS,2,2,4),complex); Rp=np.zeros((B,NS,2,4),complex); a=1/math.sqrt(NS)
  for s in range(2):
    Lp[:,:,s,1,s]=a
    Rp[:,:,s,2+s]=a
  return Lp,Rp

def overlap2(Lp,Rp):
  B=Lp.shape[0]; M=np.zeros((B,2,2),complex); a=1/math.sqrt(NS)
  for ci in range(2):
    for co in range(2):
      vals=[]
      for s in range(2):
        col=(0 if ci==0 else 2)+s
        if co==0: amp=a*np.sum(Lp[:,:,s,1,col],axis=1)
        else: amp=a*np.sum(Rp[:,:,s,col],axis=1)
        vals.append(amp)
      M[:,co,ci]=0.5*(vals[0]+vals[1])
  return M

def batch_eval(pts):
  B=len(pts); nU=len(NUM)
  th=np.zeros((B,24)); th[:,NUM]=2*np.pi*pts[:,:nU]-np.pi; u1=np.exp(1j*th)
  Q=np.zeros((B,24,4));Q[:,:,0]=1
  q=su2_from_u(pts[:,nU:].reshape(B,nU,3))
  for j,l in enumerate(NUM): Q[:,l]=q[:,j]
  Q2=qmat(Q)
  phase,K1,K2=gauge_phase(th,Q)
  Lp,Rp=init_sources(B); mats={}
  for n in range(1,11):
    Lp,Rp=step(Lp,Rp,u1,Q2)
    if n in (8,9,10): mats[n]=overlap2(Lp,Rp)
  # route/family 6x6
  T=np.zeros((B,6,6),complex)
  for a,d in enumerate(D):
    # chirality-major x family
    K=np.einsum('nij,fg->nifjg',mats[d],WF[a],optimize=True).reshape(B,6,6)
    T += (COEFF[a]/3)*K
  return phase,T,K1,K2

def rep(m,seed,batch=256):
  dim=4*len(NUM); sam=qmc.Sobol(dim,scramble=True,seed=seed); pts=sam.random_base2(m)
  tsum=np.zeros((6,6),complex); zsum=0j; tfree=np.zeros((6,6),complex); k1=0.;k2=0.
  for i in range(0,len(pts),batch):
    ph,T,a,b=batch_eval(pts[i:i+batch]); n=len(ph)
    tsum += np.einsum('n,nij->ij',ph,T,optimize=True)
    zsum += ph.sum(); tfree += T.sum(axis=0); k1+=a.sum();k2+=b.sum()
  N=len(pts); A=tsum/N; Zg=zsum/N; Acond=A/Zg if abs(Zg)>0 else A*np.nan
  sv=np.linalg.svd(A,compute_uv=False); svc=np.linalg.svd(Acond,compute_uv=False); svfree=np.linalg.svd(tfree/N,compute_uv=False)
  def kval(s):
    s=s[s>1e-14]; return float(-np.mean(np.log(s))) # -1/12 log det T*T = -1/6 sum log s_i = -mean(log s) for 6
  return {'seed':seed,'m':m,'N':N,
    'Zg':[float(Zg.real),float(Zg.imag)],'abs_Zg':float(abs(Zg)),
    'raw_T':[[[float(z.real),float(z.imag)] for z in row] for row in A],
    'raw_singular_values':sv.tolist(),'raw_K':kval(sv),
    'conditional_singular_values':svc.tolist(),'conditional_K':kval(svc),
    'unweighted_matter_singular_values':svfree.tolist(),'unweighted_matter_K':kval(svfree),
    'mean_K1':float(k1/N),'mean_K2':float(k2/N)}

def main():
  import argparse
  ap=argparse.ArgumentParser();ap.add_argument('--m',type=int,default=12);ap.add_argument('--seeds',default='11,29,47,83');a=ap.parse_args()
  reps=[]
  for s in map(int,a.seeds.split(',')):
    r=rep(a.m,s);reps.append(r);print('seed',s,'rawK',r['raw_K'],'condK',r['conditional_K'],'|Z|',r['abs_Zg'],flush=True)
  keys=['raw_K','conditional_K','unweighted_matter_K','abs_Zg']
  summary={k:{'mean':float(np.mean([r[k] for r in reps])),'sd':float(np.std([r[k] for r in reps],ddof=1))} for k in keys}
  out={'status':'L2_PRIMITIVE_HAAR_COUPLED_RANK6_STATIC_ROUTE_CONTROL','external_particle_data_used':False,
       'geometry':{'L':2,'sites':8,'links':24,'tree_links':TREE,'sampled_links':NUM,'qmc_dimension':4*len(NUM)},
       'matter':{'q_L':-3,'q_R':-6,'SU2_L':'fundamental','SU2_R':'singlet','h0':[0,1],
                 'stream':'Sx Sy Sz','route_temporal_monodromies':D,'family_Weyl':['I','X','Z'],'route_coefficients':[[float(z.real),float(z.imag)] for z in COEFF],
                 'spin_reduction':'normalized trace over the two C8 spin source states'},
       'gauge':{'phi_star':'pi/63','A_U1':A1,'A_SU2':A2,'vacuum':'primitive Haar/zero-electric one-step source'},
       'replicates':reps,'summary':summary,
       'interpretation':'This is a genuinely non-factorized single-slice primitive-vacuum gauge-matter contraction: the same U(1) and SU(2) links enter the 24-plaquette phase and the rank-six route transfer before Haar averaging. It is not the Kato-dressed repeated quantum-gauge Floquet correlator because the electric kernel between successive macro-steps is not represented by a static background integral.'}
  p=ROOT/'results'/'microscopic_closure'/'L2_COUPLED_RANK6_STATIC_ROUTE.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'summary':summary},indent=2))
if __name__=='__main__': main()
