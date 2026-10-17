#!/usr/bin/env python3
from __future__ import annotations
import numpy as np, math, time, json, hashlib
from pathlib import Path
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import jv
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'results'
PHI=math.pi/63.0
z=np.load(DATA/'S24_OPERATOR.npz'); n=len(z['KE']); KE=z['KE']
KB=csr_matrix((z['KB_data'],z['KB_indices'],z['KB_indptr']),shape=(n,n))
v0=np.load(DATA/'S24_FILTER.npz')['v']; th=float(z['prev_phase'][0]); reps=z['reps']; t=time.time()
center=rad=96.0; aa=PHI*rad
for deg in range(100):
    tail=2*sum(abs(jv(k,aa)) for k in range(deg+1,220))
    if tail<2e-13: break
X=(KB-center*identity(n,format='csr'))/rad; ph=np.exp(-1j*PHI*center)
co=[jv(0,aa)]+[2*((-1j)**k)*jv(k,aa) for k in range(1,deg+1)]
def ub(x,adj=False):
    cs=[np.conj(c) for c in co] if adj else co; p=np.conj(ph) if adj else ph
    t0=np.asarray(x,complex); y=cs[0]*t0
    if deg:
        t1=X@t0; y+=cs[1]*t1
        for kk in range(2,deg+1):
            t2=2*(X@t1)-t0; y+=cs[kk]*t2; t0,t1=t1,t2
    return p*y
ue=np.exp(-1j*PHI*KE)
U=lambda x:ub(ue*x,False)
Ud=lambda x:np.conj(ue)*ub(x,True)
a=np.exp(-1j*th)
H=LinearOperator((n,n),matvec=lambda x:2*x-a*U(x)-np.conj(a)*Ud(x),dtype=np.complex128)
vals,vecs=eigsh(H,k=1,which='SA',v0=v0,tol=2e-8,maxiter=80)
v=vecs[:,0]/np.linalg.norm(vecs[:,0]); uv=U(v); lam=np.vdot(v,uv)
e2=np.sum(reps.astype(np.int32)**2,axis=1); prob=np.abs(v)**2
sel={'H_eval':float(vals[0]),'phase':float(np.angle(lam)),'mod_rayleigh':float(abs(lam)),
     'U_residual':float(np.linalg.norm(uv-lam*v)),'overlap_prev':float(abs(np.vdot(v0,v))),
     'boundary_prob':float(prob[e2==24].sum()),'mean_E2':float(np.dot(prob,e2))}
np.savez_compressed(DATA/'U1_L2_KATO_S24_STATE.npz', reps=reps, sizes=z['sizes'], v=v, phase=np.array([float(np.angle(lam))]), KE=KE, KB_data=z['KB_data'], KB_indices=z['KB_indices'], KB_indptr=z['KB_indptr'])
out={'status':'PASS_S24_ORBIT_KATO_REFINEMENT' if sel['U_residual']<2e-12 else 'CHECK',
     'S':24,'orbit_dim':n,'prev_phase':th,'selected':sel,'delta_phase':sel['phase']-th,
     'cheb_degree':deg,'cheb_tail':tail,'seconds':time.time()-t}
p=OUT/'U1_L2_KATO_S24_ORBIT_REPRODUCED.json'; p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); print('sha256',hashlib.sha256(p.read_bytes()).hexdigest())
if out['status']!='PASS_S24_ORBIT_KATO_REFINEMENT': raise SystemExit(2)
