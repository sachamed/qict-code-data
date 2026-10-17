#!/usr/bin/env python3
from __future__ import annotations
import json,itertools
import numpy as np
P=[np.array([[1,0],[0,0]],complex),np.array([[0,0],[0,1]],complex)]
SX=np.array([[0,1],[1,0]],complex);SY=np.array([[0,-1j],[1j,0]],complex);SZ=np.array([[1,0],[0,-1]],complex)
PROJ={mu:[(np.eye(2)+s)/2,(np.eye(2)-s)/2] for mu,s in zip((0,1,2),(SX,SY,SZ))}

def randu(rng,n):
    X=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));Q,R=np.linalg.qr(X);d=np.diag(R);return Q@(np.diag(np.conj(d/np.abs(d))))

def kron3(a,b,c):return np.kron(np.kron(a,b),c)

def self_test():
    rng=np.random.default_rng(20260918); n1,n2=3,4; hops=6; mus=[2,1,0,2,1,0]
    A=[];B=[];G1=[];G2=[]
    for t in range(hops):
      A.append([randu(rng,n1),randu(rng,n1)]); B.append([randu(rng,n2),randu(rng,n2)])
      G1.append(randu(rng,n1));G2.append(randu(rng,n2))
    s=rngu=lambda n:(lambda v:v/np.linalg.norm(v))(rng.normal(size=n)+1j*rng.normal(size=n))
    vs=rngu(2);v1=rngu(n1);v2=rngu(n2);v=np.kron(np.kron(vs,v1),v2);w=v.copy()
    # Direct common Hilbert evolution: gauge layer then conditional C8 branch sum.
    for t,mu in enumerate(mus):
      w=kron3(np.eye(2),G1[t],G2[t])@w
      S=sum(kron3(PROJ[mu][sig],A[t][sig],B[t][sig]) for sig in (0,1))
      w=S@w
    direct=np.vdot(v,w)
    # Exact path-conditioned influence decomposition. Same branch history is used in both groups.
    cond=0j
    for sigs in itertools.product((0,1),repeat=hops):
      ws=vs.copy();x1=v1.copy();x2=v2.copy()
      for t,(mu,sig) in enumerate(zip(mus,sigs)):
        x1=A[t][sig]@(G1[t]@x1);x2=B[t][sig]@(G2[t]@x2);ws=PROJ[mu][sig]@ws
      cond += np.vdot(vs,ws)*np.vdot(v1,x1)*np.vdot(v2,x2)
    err=abs(direct-cond);assert err<2e-13
    # Deliberately wrong post-sum factorization for contrast.
    f1=0j;f2=0j;fs=0j
    # independent sums cannot preserve branch correlation; use normalized sums only as a diagnostic counterexample.
    for sigs in itertools.product((0,1),repeat=hops):
      ws=vs.copy();x1=v1.copy();x2=v2.copy()
      for t,(mu,sig) in enumerate(zip(mus,sigs)):
        x1=A[t][sig]@(G1[t]@x1);x2=B[t][sig]@(G2[t]@x2);ws=PROJ[mu][sig]@ws
      fs+=np.vdot(vs,ws);f1+=np.vdot(v1,x1);f2+=np.vdot(v2,x2)
    wrong=fs*f1*f2/(2**(2*hops)); gap=abs(direct-wrong)
    return {'status':'PASS_EXACT_CONDITIONAL_PRODUCT_GROUP_INFLUENCE_DECOMPOSITION','hops_tested':hops,
      'full_dimension':2*n1*n2,'branch_histories':2**hops,'direct_amplitude':[direct.real,direct.imag],
      'conditional_common_history_amplitude':[cond.real,cond.imag],'residual_abs':err,
      'independent_postsum_factorization_gap_abs':gap,
      'identity':'For each hop S_mu=sum_sigma P_mu^sigma tensor T_U1(mu,sigma) tensor T_SU2(mu,sigma). Therefore any repeated matrix element is a sum over one common branch history sigma_1...sigma_N of spin_weight * F_U1(history) * F_SU2(history). The gauge groups may be contracted separately only while this branch tensor is kept open; multiplying their already-summed route amplitudes is not equivalent.',
      'task3_consequence':'Replace the Cartesian U(1)xSU(2) frontier by two symmetry-specific influence frontiers linked by the binary C8 branch MPS. On L=2 the position variable is already eliminated, so the shared matter auxiliary is only spin/weak dimension 4 and the branch leg dimension 2 per substep.',
      'error_composition':'For normalized influence tensors, if separate certified approximations have 2-norm errors eta1 and eta2, the pathwise product error is bounded by eta1+eta2+eta1*eta2 before adding the independent Peter-Weyl tail ledger.'}
if __name__=='__main__':print(json.dumps(self_test(),indent=2))
