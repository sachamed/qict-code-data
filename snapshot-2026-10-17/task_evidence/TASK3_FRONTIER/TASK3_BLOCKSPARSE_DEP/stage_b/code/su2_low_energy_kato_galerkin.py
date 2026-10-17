#!/usr/bin/env python3
from __future__ import annotations
import sys, os, time, json, math
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm, schur, eigh
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; OUT=ROOT/'results';sys.path.insert(0,str(HERE))
import su2_k0_orbit_krylov as k
import su2_global_spin_network as gs
PHI=math.pi/63.0

def energy(st): return sum(n*(n+2) for n in st.links)

def enumerate_basis(M):
    vac=k.canonical(gs.vacuum_state()); seen={vac};Q=deque([vac]);t=time.time()
    while Q:
        st=Q.popleft()
        for pid in range(24):
            for ns,z in k.wp_action(st,pid):
                if abs(z)<1e-14: continue
                ns=k.canonical(ns)
                if energy(ns)<=M and ns not in seen:
                    seen.add(ns); Q.append(ns)
    reps=sorted(seen,key=lambda s:(s.links,s.paths))
    return reps,time.time()-t

def build(M):
    reps,enumsec=enumerate_basis(M); idx={s:i for i,s in enumerate(reps)}; n=len(reps)
    rr=[];cc=[];dd=[]; missing_weight=0.0;t=time.time()
    for a,st in enumerate(reps):
        na=len(k.orbit(st)); agg=defaultdict(complex)
        for pid in range(24):
            for ns,z in k.wp_action(st,pid):
                if abs(z)<1e-14:continue
                cr=k.canonical(ns); agg[cr]+=z
        for cr,z in agg.items():
            b=idx.get(cr)
            if b is None:
                missing_weight += abs(z)**2
                continue
            nb=len(k.orbit(cr)); rr.append(b);cc.append(a);dd.append(math.sqrt(na/nb)*z)
    H=coo_matrix((dd,(rr,cc)),shape=(n,n),dtype=complex).tocsr();H.sum_duplicates()
    herm=H-H.getH(); hres=float(np.linalg.norm(herm.data)) if herm.nnz else 0.0
    E=np.asarray([energy(s) for s in reps],float); K=diags(E,0,format='csr')-16.0*H
    vals,vecs=eigsh(K,k=min(3,max(1,n-1)),which='SA',tol=1e-12)
    order=np.argsort(vals);vals=vals[order];vecs=vecs[:,order];g=vecs[:,0]/np.linalg.norm(vecs[:,0])
    vaci=idx[k.canonical(gs.vacuum_state())]
    # Dense Floquet continuation is acceptable through M=24 (n=536).
    Hd=H.toarray(); Ed=np.diag(E)
    prev=g.copy(); minov=1.0; mingap=99.0
    for s in np.linspace(0.02,1.0,50):
        U=np.exp(-1j*s*PHI*32*24)*expm(1j*s*PHI*16*Hd)@np.diag(np.exp(-1j*s*PHI*E))
        T,Q=schur(U,output='complex');lam=np.diag(T);ov=np.abs(Q.conj().T@prev)**2;j=int(np.argmax(ov));v=Q[:,j]
        a=np.vdot(prev,v)
        if abs(a):v*=np.exp(-1j*np.angle(a))
        ph=-np.angle(lam);dp=np.abs(((ph-ph[j])+np.pi)%(2*np.pi)-np.pi);dp[j]=99
        minov=min(minov,float(ov[j]));mingap=min(mingap,float(dp.min()));prev=v
    U=np.exp(-1j*PHI*32*24)*expm(1j*PHI*16*Hd)@np.diag(np.exp(-1j*PHI*E)); lam=np.vdot(prev,U@prev); fres=float(np.linalg.norm(U@prev-lam*prev))
    prob=np.abs(prev)**2
    return {
      'M':M,'orbit_dim':n,'enumeration_seconds':enumsec,'operator_build_seconds':time.time()-t,
      'HB_nnz':int(H.nnz),'HB_hermiticity_fro_residual':hres,'omitted_transition_weight_sq_diagnostic':float(missing_weight),
      'Kprim_lowest_eigenvalues':[float(x) for x in vals],
      'Kprim_gap':float(vals[1]-vals[0]) if len(vals)>1 else None,
      'primitive_ground_vacuum_probability':float(abs(g[vaci])**2),
      'floquet_phase':float(np.angle(lam)),'floquet_modulus':float(abs(lam)),'floquet_eigen_residual':fres,
      'floquet_min_continuation_overlap':minov,'floquet_min_phase_gap':mingap,
      'floquet_vacuum_probability':float(prob[vaci]),'floquet_mean_KE':float(np.dot(prob,E)),
      'floquet_boundary_probability':float(prob[E>=M-1e-12].sum())
    }

def main():
    rows=[]
    for M in (12,18,24):
        r=build(M);rows.append(r);print('M',M,'dim',r['orbit_dim'],'phase',r['floquet_phase'],'res',r['floquet_eigen_residual'],'boundary',r['floquet_boundary_probability'],flush=True)
    out={
      'status':'PASS_SU2_LOW_ENERGY_KATO_GALERKIN_DIAGNOSTIC',
      'rows':rows,
      'phase_successive_changes':[abs(rows[i]['floquet_phase']-rows[i-1]['floquet_phase']) for i in range(1,len(rows))],
      'scope':'Translation-orbit low-electric-energy SU(2) Galerkin diagnostic. It is not the frozen Peter-Weyl PEPO regulator and cannot populate h_d. Its purpose is to quantify whether a compact Kato seed may initialize the boundary-MPS solver.'
    }
    (OUT/'SU2_LOW_ENERGY_KATO_GALERKIN.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
