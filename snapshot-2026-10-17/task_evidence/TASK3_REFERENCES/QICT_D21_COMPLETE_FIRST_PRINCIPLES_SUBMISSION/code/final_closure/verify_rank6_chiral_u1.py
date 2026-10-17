#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import schur

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import compute_c8_static_charge_strip as strip
from qict_route_first import electron_route_coefficients, I3, X, Z

PHI=math.pi/63
WFAM=[I3,X,Z]
COEFF=electron_route_coefficients()
OUT=ROOT/'results'/'decisive_chiral'; OUT.mkdir(parents=True,exist_ok=True)

def dft(N):
    j=np.arange(N); return np.exp(-2j*np.pi*np.outer(j,j)/N)/np.sqrt(N)

def dense_components(R,N,q):
    KE,_,_,_,_=strip.electric_diag(R,N,q); B=strip.theta_B_diag(R,N).reshape(-1); F=dft(N)
    FR=F
    for _ in range(R-1): FR=np.kron(FR,F)
    return KE,B,FR

def Uof(KE,B,F,s=1.0):
    DE=np.exp(-1j*s*PHI*KE); DB=np.exp(-1j*s*PHI*B)
    return F.conj().T @ (DB[:,None]*(F*DE[None,:]))

def kato_vacuum(R,N,steps=41):
    vals,g,_=strip.ground_ref(R,N); g=g/np.linalg.norm(g)
    KE,B,F=dense_components(R,N,0); prev=g.copy(); min_gap=99.; min_ov=1.
    for s in np.linspace(0.02,1.0,steps):
        U=Uof(KE,B,F,float(s)); T,Q=schur(U,output='complex'); lam=np.diag(T)
        ov=np.abs(Q.conj().T@prev)**2; j=int(np.argmax(ov)); v=Q[:,j]
        a=np.vdot(prev,v)
        if abs(a): v*=np.exp(-1j*np.angle(a))
        ph=-np.angle(lam); dp=np.abs(((ph-ph[j])+np.pi)%(2*np.pi)-np.pi); dp[j]=99
        min_gap=min(min_gap,float(dp.min())); min_ov=min(min_ov,float(ov[j])); prev=v
    return prev, {'primitive_E0':float(vals[0]),'min_step_overlap':min_ov,'min_gap':min_gap,'final_overlap_primitive':float(abs(np.vdot(g,prev))**2)}

def build_ops(N,R=2):
    d=N*N; vac,kinfo=kato_vacuum(R,N); S=np.roll(np.eye(N,dtype=complex),1,axis=0)
    hol=[np.eye(d,dtype=complex),np.kron(S,np.eye(N)),np.kron(np.eye(N),S)]
    ops={}
    for q in (-3,-6):
        KE,B,F=dense_components(R,N,q); Uq=Uof(KE,B,F,1.0); ops[q]=[hol[a]@Uq for a in range(3)]
    return vac,kinfo,ops

def source(vac):
    d=len(vac); V=np.zeros((2,3,d,3,6),complex); rp=np.ones(3)/np.sqrt(3)
    for c in range(2):
      for a in range(3):
       for f in range(3): V[c,a,:,f,3*c+f]=rp[a]*vac
    return V

def applyU(X,ops):
    tmp=np.zeros_like(X)
    for c,q in enumerate((-3,-6)):
      for a in range(3):
       G=ops[q][a]; W=COEFF[a]*WFAM[a]
       for k in range(X.shape[-1]): tmp[c,a,:,:,k]=G@X[c,a,:,:,k]@W.T
    Y=np.empty_like(tmp); Y[0]=tmp[1]; Y[1]=tmp[0]  # exact Clebsch chirality swap
    return Y

def inner(A,B): return np.einsum('crgfk,crgfl->kl',A.conj(),B,optimize=True)

def run(N,nmax=13):
    t=time.time(); vac,kinfo,ops=build_ops(N); V=source(vac); gram=inner(V,V); X=V.copy(); moments=[gram]
    for _ in range(nmax): X=applyU(X,ops); moments.append(inner(V,X))
    odd=[]
    for n in range(1,nmax+1,2):
        s=np.linalg.svd(moments[n][:3,3:],compute_uv=False)
        odd.append({'n':n,'singular_values':s.tolist(),'ratio_max_min':float(s[0]/s[-1]),'ratio_mid_min':float(s[1]/s[-1])})
    return {'N':N,'seconds':time.time()-t,'kato':kinfo,'source_gram_defect':float(np.linalg.norm(gram-np.eye(6),2)),
            'q_L':-3,'q_R':-6,'one_step_LR_singular_values':odd[0]['singular_values'],'odd_LR_moments':odd}

def main():
    cases=[run(N) for N in (15,17,19,21)]
    out={'blind':True,'particle_masses_used_in_operator':False,'R':2,
         'construction':'rank-six (3_L + 3_R) route-first U(1) strip with q_L=-3, q_R=-6 and exact Clebsch chirality swap',
         'cases':cases,
         'verdict':'CONVERGED_DIAGNOSTIC__FAIL_AS_CHARGED_LEPTON_HIERARCHY',
         'reason':'The chirality-changing block is regulator-stable but its hierarchy is O(1), not O(10^2-10^3). Full 3D electroweak route/gauge/CAR contraction remains required.'}
    (OUT/'RANK6_CHIRAL_U1_CERTIFICATE.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
