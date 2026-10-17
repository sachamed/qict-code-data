#!/usr/bin/env python3
from __future__ import annotations
import json, math, importlib.util
from pathlib import Path
import numpy as np
from scipy.sparse import diags, identity

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/'results'/'LOCAL_ELECTRIC_DEFECT_BLOCKS.json'
PHI=math.pi/63.0

def load_su2():
    p=ROOT/'upstream_task2_code'/'compute_su2_full_link_matter_route.py'
    spec=importlib.util.spec_from_file_location('su2full',p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def u1_block(q=-3,emax=340):
    E=np.arange(-emax,emax+1,dtype=int)
    P=lambda x:2*x*x+x
    delta=np.array([P(int(e+q))-P(int(e)) for e in E],float)
    z=np.exp(-1j*PHI*delta)
    # Orientation reversal is q -> -q in the same fixed link coordinate.
    dm=np.array([P(int(e-q))-P(int(e)) for e in E],float)
    zm=np.exp(-1j*PHI*dm)
    return {
      'q':q,'E_range':[-emax,emax],'dimension':len(E),
      'formula_plus':'exp[-i phi (P_+(E+q)-P_+(E))]',
      'formula_minus':'exp[-i phi (P_+(E-q)-P_+(E))]',
      'max_unit_modulus_residual_plus':float(np.max(np.abs(np.abs(z)-1))),
      'max_unit_modulus_residual_minus':float(np.max(np.abs(np.abs(zm)-1))),
      'phase_increment_is_affine_in_E':True,
      'deltaP_plus_coefficients':{'E':4*q,'constant':2*q*q+q},
      'deltaP_minus_coefficients':{'E':-4*q,'constant':2*q*q-q},
    }

def fro(A):
    return float(np.linalg.norm(A.data)) if getattr(A,'nnz',0) else 0.0

def su2_case(nmax):
    m=load_su2(); st,idx,W,Mchi,ke,KB=m.operators(nmax); G=len(st); J=2*G
    phase=np.repeat(np.exp(-1j*PHI*ke),2)
    UE=diags(phase,0,shape=(J,J),format='csr',dtype=complex)
    # Plus orientation: W is multiplication by the fundamental transporter.
    Dp=(W.conj().T@UE@W@UE.conj().T).tocsr()
    # Minus orientation uses inverse transporter W^dag in the conjugation.
    Dm=(W@UE@W.conj().T@UE.conj().T).tocsr()
    # To suppress finite-Peter-Weyl boundary artifacts, test columns n<=nmax-2.
    safe=[]
    for i,(n,mm,rr) in enumerate(st):
        if n<=nmax-2: safe.extend([2*i,2*i+1])
    def unit_res(D):
        Ds=D[:,safe]
        A=(Ds.conj().T@Ds-identity(len(safe),format='csr',dtype=complex)).tocsr()
        return fro(A)
    def nontrivial(D):
        A=(D[:,safe]-identity(J,format='csr',dtype=complex)[:,safe]).tocsr()
        return fro(A)
    # Matter-spin covariance: simultaneous change of basis on the two matter
    # indices leaves singular values. Record local sparsity and safe unitarity.
    return {
      'twice_j_max':nmax,'gauge_dim':G,'joint_dim':J,'safe_joint_columns':len(safe),
      'plus_nnz':int(Dp.nnz),'minus_nnz':int(Dm.nnz),
      'plus_safe_isometry_fro_residual':unit_res(Dp),
      'minus_safe_isometry_fro_residual':unit_res(Dm),
      'plus_nontrivial_fro_safe':nontrivial(Dp),
      'minus_nontrivial_fro_safe':nontrivial(Dm),
      'definition_plus':'W^dag exp(-i phi K_E) W exp(+i phi K_E)',
      'definition_minus':'W exp(-i phi K_E) W^dag exp(+i phi K_E)',
      'boundary_rule':'Columns with n>nmax-2 are excluded from the algebraic unitarity regression; their physical contribution is controlled by the independent Peter-Weyl tail ledger.'
    }

def main():
    u=u1_block()
    cases=[su2_case(12),su2_case(14)]
    checks={
      'u1_unitary':u['max_unit_modulus_residual_plus']<1e-14 and u['max_unit_modulus_residual_minus']<1e-14,
      'su2_coarse_safe_unitarity':max(cases[0]['plus_safe_isometry_fro_residual'],cases[0]['minus_safe_isometry_fro_residual'])<2e-11,
      'su2_fine_safe_unitarity':max(cases[1]['plus_safe_isometry_fro_residual'],cases[1]['minus_safe_isometry_fro_residual'])<3e-11,
      'defects_nontrivial':min(cases[0]['plus_nontrivial_fro_safe'],cases[0]['minus_nontrivial_fro_safe'])>1e-3,
    }
    out={
      'status':'PASS_COARSE_FINE_LOCAL_ELECTRIC_DEFECT_BLOCKS' if all(checks.values()) else 'FAIL',
      'phi':PHI,'u1':u,'su2':cases,'checks':checks,
      'frontier_use':'These are the branch-conditioned unary link tensors inserted on the twelve C8-active links after the exact relative-defect transformation. They do not replace the neutral magnetic/Kato environment and do not by themselves evaluate h_d.'
    }
    OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
