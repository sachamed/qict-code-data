#!/usr/bin/env python3
from __future__ import annotations
import json, math, importlib.util
from pathlib import Path
import numpy as np
from scipy.sparse import identity, kron, diags

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/'results'/'RELATIVE_ELECTRIC_DEFECT_REDUCTION.json'
PHI=math.pi/63.0

def u1_test(M=12,q=-3):
    es=np.arange(-M,M+1)
    dim=len(es); idx={int(e):i for i,e in enumerate(es)}
    # D_q |E> = |E+q>, tested only on safe interior columns.
    D=np.zeros((dim,dim),complex)
    safe=[]
    for e in es:
        j=idx.get(int(e+q))
        if j is not None:
            D[j,idx[int(e)]]=1
            if -M+abs(q) <= e <= M-abs(q): safe.append(idx[int(e)])
    p=lambda E:2*E*E+E
    UE=np.diag(np.exp(-1j*PHI*np.array([p(int(e)) for e in es],float)))
    # Conjugation identity on columns that remain inside cutoff.
    lhs=D.conj().T@UE@D
    target=np.zeros_like(lhs)
    for e in es:
        if int(e+q) in idx: target[idx[int(e)],idx[int(e)]]=np.exp(-1j*PHI*p(int(e+q)))
    r_conj=float(np.linalg.norm((lhs-target)[:,safe],2)) if safe else 0.0
    # Plaquette/closed-loop character shifts commute with open-line shift in U(1).
    P=np.zeros((dim,dim),complex)
    for e in es:
        for s in (-1,1):
            j=idx.get(int(e+s))
            if j is not None:P[j,idx[int(e)]]+=1
    comm=P@D-D@P
    safe2=[idx[int(e)] for e in es if int(e+q) in idx and int(e+1) in idx and int(e-1) in idx and int(e+q+1) in idx and int(e+q-1) in idx]
    r_comm=float(np.linalg.norm(comm[:,safe2],2)) if safe2 else 0.0
    defect=lhs@UE.conj().T
    return {'M':M,'q':q,'conjugation_residual_safe':r_conj,'magnetic_shift_commutator_residual_safe':r_comm,
            'defect_nontrivial_2norm_safe':float(np.linalg.norm((defect-np.eye(dim))[:,safe],2)) if safe else 0.0,
            'identity':'D_q^dag f(E) D_q = f(E+q) on the untruncated flux lattice; closed character shifts commute with D_q.'}

def load_task2_su2():
    p=ROOT/'upstream_task2_code'/'compute_su2_full_link_matter_route.py'
    spec=importlib.util.spec_from_file_location('su2one',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def su2_test(nmax=8):
    m=load_task2_su2(); st,idx,W,Mchi,ke,KB=m.operators(nmax);G=len(st)
    Mj=kron(Mchi,identity(2,format='csr',dtype=complex),format='csr')
    comm=(Mj@W-W@Mj).tocsr()
    # safe input columns n<=nmax-2: both a fundamental multiplication and a character multiplication
    # stay inside the finite Peter-Weyl cutoff, so the exact multiplication commutator is visible.
    safe=[]
    for i,(n,mm,rr) in enumerate(st):
        if n<=nmax-2:safe.extend([2*i,2*i+1])
    C=comm[:,safe]
    rcomm=float(np.linalg.norm(C.data)) if C.nnz else 0.0
    UE=diags(np.repeat(np.exp(-1j*PHI*ke),2),0,format='csr')
    # On safe interior, W is an isometry; relative electric defect is W^dag UE W UE^dag.
    Ws=W[:,safe]
    gram=(Ws.conj().T@Ws-identity(len(safe),format='csr')).tocsr()
    riso=float(np.linalg.norm(gram.data)) if gram.nnz else 0.0
    defect=(Ws.conj().T@UE@Ws) @ UE[safe,:][:,safe].conj().T
    I=identity(len(safe),format='csr',dtype=complex)
    dd=(defect-I).tocsr()
    dn=float(np.linalg.norm(dd.data)) if dd.nnz else 0.0
    return {'nmax':nmax,'gauge_dim':G,'joint_dim':2*G,'safe_joint_columns':len(safe),
            'magnetic_character_commutator_fro_residual_safe':rcomm,
            'fundamental_transporter_isometry_fro_residual_safe':riso,
            'relative_electric_defect_fro_norm_safe':dn,
            'identity':'[M_chi,W_fund]=0 exactly before cutoff because both are multiplication operators on SU(2); all charged/non-neutral difference can therefore be moved to conjugated electric kernels and matter-route updates.'}

def main():
    u=u1_test(); s=su2_test()
    checks={'u1_exact_conjugation':u['conjugation_residual_safe']<1e-13,
            'u1_magnetic_commutation':u['magnetic_shift_commutator_residual_safe']<1e-13,
            'su2_magnetic_commutation':s['magnetic_character_commutator_fro_residual_safe']<2e-12,
            'su2_transporter_isometry':s['fundamental_transporter_isometry_fro_residual_safe']<2e-11,
            'electric_defect_nontrivial':u['defect_nontrivial_2norm_safe']>1e-3 and s['relative_electric_defect_fro_norm_safe']>1e-3}
    out={'status':'PASS_RELATIVE_ELECTRIC_DEFECT_LOCALIZATION' if all(checks.values()) else 'FAIL',
         'theorem':'For a fixed Wilson dressing/worldline multiplication operator D, every plaquette magnetic multiplication operator commutes with D. In the relative charged correlator, moving D through a gauge macro-step therefore leaves the magnetic factor unchanged and replaces only the electric factor by D^dag exp(-i phi K_E) D. The defect is supported on links touched by D. Alternative matter paths must still be recombined coherently; this theorem is not a factorization over paths.',
         'u1':u,'su2':s,'checks':checks,
         'consequence_for_pepo':'Use the neutral Kato gauge frontier as the environment. Insert branch-conditioned local electric-defect tensors only on the dressing/worldline links, keep the common C8 branch index open, and retain magnetic plaquette evolution in the neutral frontier. This removes the need for a separately propagated global charged gauge environment but does not by itself evaluate h_8,h_9,h_10.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
