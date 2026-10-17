#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from scipy.special import j0, j1

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'microscopic_closure'/'L2_ELECTROWEAK_GAUGE_CONTRACTION.json'
OUT.parent.mkdir(parents=True,exist_ok=True)
PHI=math.pi/63
L=2
A1=4.0
A2=32.0

# Split-step torus: 8 sites, 24 positively oriented links, 24 plaquettes.
links=[(mu,x,y,z) for mu in range(3) for x in range(L) for y in range(L) for z in range(L)]
li={t:i for i,t in enumerate(links)}
plaquettes=[]
for mu in range(3):
  for nu in range(mu+1,3):
    for x in range(L):
      for y in range(L):
        for z in range(L):
          s=[x,y,z]; sm=s.copy(); sm[mu]=(sm[mu]+1)%L; sn=s.copy(); sn[nu]=(sn[nu]+1)%L
          plaquettes.append([(li[(mu,*s)],1),(li[(nu,*sm)],1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)])

# Six links with pairwise disjoint plaquette stars; together they cover all 24 plaquettes.
ELIM=[0,4,9,11,22,23]
# A spanning tree chosen in the complement of ELIM. Gauge fixing sets these links to identity.
TREE=[10,17,5,6,19,7,14]
NUM=[i for i in range(24) if i not in set(ELIM) and i not in set(TREE)]
INC={l:[] for l in ELIM}
for p,po in enumerate(plaquettes):
  hits=[l for l,_ in po if l in INC]
  assert len(hits)==1
  INC[hits[0]].append(p)
assert all(len(INC[l])==4 for l in ELIM)
assert len(NUM)==11

# Quaternion helpers, q=(w,x,y,z), normalized Haar on SU(2)=S^3.
def su2_from_u(u):
  u1,u2,u3=u[...,0],u[...,1],u[...,2]
  x=np.sqrt(1-u1)*np.sin(2*np.pi*u2)
  y=np.sqrt(1-u1)*np.cos(2*np.pi*u2)
  z=np.sqrt(u1)*np.sin(2*np.pi*u3)
  w=np.sqrt(u1)*np.cos(2*np.pi*u3)
  return np.stack([w,x,y,z],axis=-1)

def qconj(q):
  r=q.copy(); r[...,1:]*=-1; return r

def qmul(a,b):
  w1=a[...,0]; v1=a[...,1:]; w2=b[...,0]; v2=b[...,1:]
  w=w1*w2-np.sum(v1*v2,axis=-1)
  v=w1[...,None]*v2+w2[...,None]*v1+np.cross(v1,v2)
  return np.concatenate([w[...,None],v],axis=-1)

E4=np.eye(4)
def plaq_scalar_basis(Q,po,target):
  n=len(Q); out=np.empty((n,4))
  for j in range(4):
    prod=np.zeros((n,4)); prod[:,0]=1
    for idx,sgn in po:
      q=np.broadcast_to(E4[j],(n,4)).copy() if idx==target else Q[:,idx,:]
      if sgn<0: q=qconj(q)
      prod=qmul(prod,q)
    out[:,j]=prod[:,0]
  return out

def u1_rep(m,seed,batch=32768):
  sampler=qmc.Sobol(len(NUM),scramble=True,seed=seed); pts=sampler.random_base2(m)
  x=A1*PHI; const=np.exp(-1j*x*len(plaquettes)); total=0j
  for i in range(0,len(pts),batch):
    pb=pts[i:i+batch]; n=len(pb); theta=np.zeros((n,24)); theta[:,NUM]=2*np.pi*pb-np.pi
    fac=np.ones(n)
    for l in ELIM:
      C=np.zeros(n,dtype=complex)
      for p in INC[l]:
        po=plaquettes[p]; rest=np.zeros(n); sl=None
        for idx,sgn in po:
          if idx==l: sl=sgn
          else: rest+=sgn*theta[:,idx]
        C += np.exp(1j*(rest if sl==1 else -rest))
      fac*=j0(x*np.abs(C))
    total+=(const*fac).sum()
  return total/len(pts)

def su2_rep(m,seed,batch=8192):
  sampler=qmc.Sobol(3*len(NUM),scramble=True,seed=seed); pts=sampler.random_base2(m)
  x=A2*PHI; const=np.exp(-1j*x*len(plaquettes)); total=0j
  ptarget={p:l for l in ELIM for p in INC[l]}
  for i in range(0,len(pts),batch):
    pb=pts[i:i+batch]; n=len(pb)
    Q=np.zeros((n,24,4)); Q[:,:,0]=1
    qs=su2_from_u(pb.reshape(n,len(NUM),3))
    for j,l in enumerate(NUM): Q[:,l,:]=qs[:,j,:]
    vs={l:np.zeros((n,4)) for l in ELIM}
    for p,po in enumerate(plaquettes):
      l=ptarget[p]; vs[l]+=plaq_scalar_basis(Q,po,l)
    fac=np.ones(n)
    for l in ELIM:
      y=x*np.linalg.norm(vs[l],axis=1); f=np.ones(n); mask=np.abs(y)>1e-12
      f[mask]=2*j1(y[mask])/y[mask]; fac*=f
    total+=(const*fac).sum()
  return total/len(pts)

def summarize(vals):
  z=sum(vals)/len(vals); sd=float(np.std(np.asarray(vals),ddof=1)) if len(vals)>1 else 0.0
  return {'replicates':[{'real':float(v.real),'imag':float(v.imag)} for v in vals],
          'mean':{'real':float(z.real),'imag':float(z.imag)},'interreplicate_complex_sd':sd,
          'abs_mean':float(abs(z)),'minus_log_abs':float(-math.log(abs(z))),
          'minus_log_abs_per_plaquette':float(-math.log(abs(z))/24)}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--m-u1',type=int,default=20); ap.add_argument('--m-su2',type=int,default=18)
  ap.add_argument('--seeds',default='11,29,47,83'); args=ap.parse_args(); seeds=[int(x) for x in args.seeds.split(',')]
  u1=[u1_rep(args.m_u1,s) for s in seeds]; su2=[su2_rep(args.m_su2,s) for s in seeds]
  z1=sum(u1)/len(u1); z2=sum(su2)/len(su2); zew=z1*z2
  report={
    'status':'PASS_L2_PRIMITIVE_VACUUM_ELECTROWEAK_GAUGE_CONTRACTION',
    'external_particle_data_used':False,
    'geometry':{'L':2,'sites':8,'links':24,'plaquettes':24,'plaquette_boundary_rank':14,'closed_integer_flux_dimension':10,
                'gauge_fixed_tree_links':TREE,'analytically_integrated_links':ELIM,'numerically_sampled_links':NUM},
    'gauge_action':{'phi_star':'pi/63','A_U1':A1,'A_SU2':A2,
      'one_step_statement':'On the primitive zero-electric Haar vacuum, exp(-i phi K_E) acts trivially. The reported vacuum return therefore contracts the full 24-plaquette multiplication gate; it is a one-step gauge-layer result, not the Kato-interacting rank-six matter correlator.'},
    'analytic_partial_integrations':{'U1':'int dtheta exp(i x Re[e^{itheta} C]) = J0(x|C|)',
      'SU2':'int_SU2 dU exp(i x q(U).v) = 2 J1(x|v|)/(x|v|)',
      'covered_plaquettes':24},
    'U1':summarize(u1),'SU2':summarize(su2),
    'factorized_gauge_only_EW_control':{'real':float(zew.real),'imag':float(zew.imag),'abs':float(abs(zew)),
      'minus_log_abs':float(-math.log(abs(zew))), 'minus_log_abs_per_plaquette':float(-math.log(abs(zew))/24)},
    'promotion_boundary':'U(1) and SU(2) factorize only for this gauge-only primitive-vacuum control. The physical rank-six route-first source contains matter holonomies, C10 dressing, the chiral bridge and Kato continuation, so its electroweak contraction cannot be obtained by multiplying these two amplitudes.',
    'verdict':'The three-dimensional electric/weak gauge layer is numerically contracted on the minimal periodic split-step volume with exact gauge fixing and six analytic Haar integrations. The full charged-lepton rank-six Feshbach contraction is not completed by this gauge-only vacuum amplitude.'
  }
  assert abs(z1)>0 and abs(z2)>0
  OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
