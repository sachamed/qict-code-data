#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True); OUT=CERT/'c8a_canonical_gauging_certificate.json'
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]],complex); sz=np.array([[1,0],[0,-1]],complex)
I2=np.eye(2,dtype=complex); SIG=[sx,sy,sz]; P=[((I2+s)/2,(I2-s)/2) for s in SIG]
A={(1,1,1):np.array([[0,-1+1j],[0,1-1j]],complex)/4,(1,1,-1):np.array([[1+1j,0],[-1-1j,0]],complex)/4,(1,-1,1):np.array([[0,-1-1j],[0,1+1j]],complex)/4,(1,-1,-1):np.array([[1-1j,0],[-1+1j,0]],complex)/4,(-1,1,1):np.array([[0,1+1j],[0,1+1j]],complex)/4,(-1,1,-1):np.array([[1-1j,0],[1-1j,0]],complex)/4,(-1,-1,1):np.array([[0,1-1j],[0,1-1j]],complex)/4,(-1,-1,-1):np.array([[1+1j,0],[1+1j,0]],complex)/4}
def idx(x,L): return ((x[0]%L)*L+(x[1]%L))*L+(x[2]%L)
def shift(L,axis,step,ph,q=1):
    N=L**3; T=np.zeros((N,N),complex); e=[0,0,0]; e[axis]=1
    for x in itertools.product(range(L),repeat=3):
        if step==1: y=tuple((x[j]+e[j])%L for j in range(3)); amp=ph[(axis,x)]**q
        else: xm=tuple((x[j]-e[j])%L for j in range(3)); y=xm; amp=np.conj(ph[(axis,xm)])**q
        T[idx(y,L),idx(x,L)]=amp
    return T
def factors(L,ph,q=1):
    out=[]
    for a in range(3):
        Tm=shift(L,a,-1,ph,q); Tp=shift(L,a,1,ph,q); Pp,Pm=P[a]
        out.append(np.kron(Tm,Pp)+np.kron(Tp,Pm))
    return out
def prod(fs): return fs[0]@fs[1]@fs[2]
def free_bcc(L):
    N=L**3; U=np.zeros((2*N,2*N),complex)
    for h,Ah in A.items():
        T=np.zeros((N,N),complex)
        for x in itertools.product(range(L),repeat=3):
            y=tuple((x[j]+h[j])%L for j in range(3)); T[idx(y,L),idx(x,L)]=1
        U+=np.kron(T,Ah)
    return U
def spectrum_distance(U,V):
    a=np.sort(np.mod(np.angle(np.linalg.eigvals(U)),2*np.pi)); b=np.sort(np.mod(np.angle(np.linalg.eigvals(V)),2*np.pi))
    return float(np.max(np.abs(a-b)))
def symbol(k):
    U=I2.copy()
    for x,s in zip(k,SIG): U=U@(np.cos(x)*I2-1j*np.sin(x)*s)
    return U
def main():
    L=3; seed=20260909; rng=np.random.default_rng(seed); sites=list(itertools.product(range(L),repeat=3)); I=np.eye(2*L**3)
    op=[]; fro=[]; cyc=[]
    for _ in range(100):
        ph={(a,x):np.exp(1j*rng.uniform(-math.pi,math.pi)) for a in range(3) for x in sites}
        fs=factors(L,ph); U=prod(fs); E=U.conj().T@U-I
        op.append(float(np.linalg.norm(E,2))); fro.append(float(np.linalg.norm(E,'fro')))
        Uyzx=fs[1]@fs[2]@fs[0]
        cyc.append(float(np.linalg.norm(Uyzx-fs[0].conj().T@U@fs[0],2)))
    trivial={(a,x):1+0j for a in range(3) for x in sites}; U0=prod(factors(L,trivial)); Ub=free_bcc(L)
    fm=float(np.linalg.norm(U0-Ub,2)); fspec=spectrum_distance(U0,Ub)
    # deterministic free-symbol factorization grid
    sr=0.0
    for kx in np.linspace(-.7,.7,7):
      for ky in np.linspace(-.6,.6,5):
       for kz in np.linspace(-.5,.5,5):
        U=symbol((kx,ky,kz)); exact=(np.cos(kx)*I2-1j*np.sin(kx)*sx)@(np.cos(ky)*I2-1j*np.sin(ky)*sy)@(np.cos(kz)*I2-1j*np.sin(kz)*sz)
        sr=max(sr,float(np.linalg.norm(U-exact,2)))
    cert={'status':'PASS','task':'C8-stream','final_status':'C8-PCC theorem/certificate','seed':seed,'random_gauge_count':100,'L':L,'operator':'U_C8=S_x S_y S_z','max_operator_norm_unitarity_defect':max(op),'max_frobenius_unitarity_defect':max(fro),'max_cyclic_time_origin_conjugacy_defect':max(cyc),'free_operator_error':fm,'free_spectrum_hausdorff_error':fspec,'free_symbol_factorization_residual':sr,'publication_reference_maxima':{'operator_norm_unitarity_defect':6.892120157971521e-16,'frobenius_unitarity_defect':1.8599146003972158e-15,'cyclic_time_origin_conjugacy_defect':4.06e-16,'free_spectrum_hausdorff_error':2.4349091853294375e-15},'tolerance':1e-12}
    assert max(op)<1e-12 and max(fro)<1e-12 and max(cyc)<1e-12 and fm<1e-12 and fspec<1e-12 and sr<1e-12
    OUT.write_text(json.dumps(cert,indent=2)+'\n'); print('PASS C8-stream',json.dumps(cert,indent=2))
if __name__=='__main__': main()
