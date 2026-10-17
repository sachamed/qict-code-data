from __future__ import annotations
import importlib.util,itertools,math,time,json,argparse
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,identity
from scipy.sparse.linalg import LinearOperator,eigsh
from scipy.special import jv
spec=importlib.util.spec_from_file_location('b','/mnt/data/task2_continue/u1_l2_affine_pair_basis_probe.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
PHI=math.pi/63; PERMS=list(itertools.permutations(range(3))); links=b.links;li=b.li

def transform_perm(E,perm):
 F=np.zeros(24,dtype=np.int16);inv=[perm.index(i) for i in range(3)]
 for old,(mu,x,y,z) in enumerate(links):
  c=(x,y,z);cp=[c[inv[i]] for i in range(3)];F[li[(perm[mu],*cp)]]=E[old]
 return tuple(map(int,F))

def build_sparse(M):
 E0,states=b.basis(M); arr=np.asarray(sorted(states),dtype=np.int16); aset={tuple(map(int,E)) for E in arr}; idx={tuple(map(int,E)):i for i,E in enumerate(arr)}
 er={};ro={}
 for e in sorted(aset):
  if e in er:continue
  A=np.asarray(e,dtype=np.int16);orb={transform_perm(A,p) for p in PERMS};orb &= aset;r=min(orb)
  for q in orb:er[q]=r
  ro[r]=sorted(orb)
 reps=sorted(ro);oi={r:i for i,r in enumerate(reps)};sizes=np.array([len(ro[r]) for r in reps],float);n=len(reps)
 KEfull=np.array([sum(b.pplus(int(e)) for e in E) for E in arr],float);KE=np.zeros(n)
 for a,r in enumerate(reps):
  vals={KEfull[idx[e]] for e in ro[r]}; assert len(vals)==1; KE[a]=next(iter(vals))
 rows=[];cols=[];data=[]
 # diagonal 96
 for a in range(n): rows.append(a);cols.append(a);data.append(96.0)
 for i,E in enumerate(arr):
  ei=tuple(map(int,E));a=oi[er[ei]]
  for pl in b.plaquettes:
   for sg in (-1,1):
    ej=tuple(map(int,E+sg*pl));j=idx.get(ej)
    if j is None:continue
    bb=oi[er[ej]];rows.append(a);cols.append(bb);data.append(-2.0/math.sqrt(sizes[a]*sizes[bb]))
 KB=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr();KB.sum_duplicates()
 # symmetrize tiny combinatorial roundoff only
 KB=(KB+KB.T)*0.5
 kmin=float(KE.min()); minidx=np.where(np.isclose(KE,kmin))[0];v0=np.zeros(n,complex)
 for j in minidx:v0[j]=math.sqrt(sizes[j])
 v0/=np.linalg.norm(v0)
 return arr,reps,sizes,KB,KE,v0,kmin

def cheb_degree(a,tol=1e-13):
 for K in range(1,80):
  # conservative finite remainder plus monotone tail estimate direct to 200
  tail=2*sum(abs(jv(n,a)) for n in range(K+1,201))
  if tail<tol:return K,tail
 raise RuntimeError('degree')

def make_ub(KB):
 n=KB.shape[0];center=96.0;rad=96.0;a=PHI*rad;K,tail=cheb_degree(a)
 X=(KB-center*identity(n,format='csr'))*(1.0/rad)
 phase=np.exp(-1j*PHI*center)
 coeff=[jv(0,a)]+[2*((-1j)**k)*jv(k,a) for k in range(1,K+1)]
 def apply(x,adj=False):
  cs=[np.conj(c) for c in coeff] if adj else coeff; ph=np.conj(phase) if adj else phase
  t0=x.astype(complex,copy=False);y=cs[0]*t0
  if K==0:return ph*y
  t1=X@t0;y=y+cs[1]*t1
  for k in range(2,K+1):
   t2=2*(X@t1)-t0;y=y+cs[k]*t2;t0,t1=t1,t2
  return ph*y
 return apply,K,tail

def run(M,theta_target):
 t=time.time();arr,reps,sizes,KB,KE,v0,kmin=build_sparse(M);ub,K,tail=make_ub(KB);ue=np.exp(-1j*PHI*KE);n=len(reps);a=np.exp(-1j*theta_target)
 def U(x):return ub(ue*x,False)
 def Ud(x):return np.conj(ue)*ub(x,True)
 def Hmv(x):return 2*x-a*U(x)-np.conj(a)*Ud(x)
 H=LinearOperator((n,n),matvec=Hmv,dtype=np.complex128)
 vals,vecs=eigsh(H,k=4,which='SA',v0=v0,tol=2e-11,maxiter=5000)
 # choose eigenvector whose Rayleigh phase is closest target
 cands=[]
 for j in range(len(vals)):
  v=vecs[:,j];v/=np.linalg.norm(v);uv=U(v);lam=np.vdot(v,uv);ph=float(np.angle(lam));dist=abs(np.angle(np.exp(1j*(ph-theta_target))));res=float(np.linalg.norm(uv-lam*v));cands.append((dist,j,ph,float(abs(lam)),res,float(vals[j])))
 cands.sort();dist,j,ph,mod,res,hval=cands[0];v=vecs[:,j]/np.linalg.norm(vecs[:,j]);
 hs=np.sort(vals.real);hgap=float(hs[1]-hs[0])
 out={'status':'PASS_AFFINE_PAIR_KATO_SPARSE' if res<1e-8 else 'FAIL','M':M,'full_basis_dim':len(arr),'sym_basis_dim':n,'KE_min':kmin,'target_phase':theta_target,'phase':ph,'phase_distance_to_target':dist,'mod_rayleigh':mod,'eigen_residual':res,'shifted_H_lowest':list(map(float,hs)),'shifted_H_gap01':hgap,'chebyshev_degree':K,'chebyshev_tail_bound':tail,'KB_hermiticity_nnz':int((KB-KB.T).nnz),'seconds':time.time()-t}
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('M',type=int);ap.add_argument('theta',type=float);a=ap.parse_args();print(json.dumps(run(a.M,a.theta),indent=2))
