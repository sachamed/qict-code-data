#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'global_sm_extension'; OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data'/'g3_renormalized_weak_vertices.json'

def polar(A):
    H=(A.conj().T@A+A.conj().T@A)/2
    w,V=np.linalg.eigh(H)
    if np.min(w)<=0: raise ValueError('rank-deficient vertex')
    return A@(V*(1/np.sqrt(w)))@V.conj().T

def invs(U):
    s13=abs(U[0,2]); c13=math.sqrt(max(0,1-s13*s13))
    s12=abs(U[0,1])/c13; s23=abs(U[1,2])/c13
    c12=math.sqrt(max(0,1-s12*s12)); c23=math.sqrt(max(0,1-s23*s23))
    J=float(np.imag(U[0,0]*U[1,1]*np.conj(U[0,1])*np.conj(U[1,0])))
    den=max(1e-300,s12*c12*s23*c23*s13*c13*c13)
    sind=max(-1,min(1,J/den)) if s13>1e-15 else 0.0
    return {'theta12_deg':math.degrees(math.asin(s12)),'theta13_deg':math.degrees(math.asin(s13)),'theta23_deg':math.degrees(math.asin(s23)),'J':J,'sin_delta_from_J':sind,'unitarity_2norm':float(np.linalg.norm(U.conj().T@U-np.eye(3),2))}

TBM=np.array([[math.sqrt(2/3),1/math.sqrt(3),0],[-1/math.sqrt(6),1/math.sqrt(3),1/math.sqrt(2)],[1/math.sqrt(6),-1/math.sqrt(3),1/math.sqrt(2)]],complex)
finite_reference=invs(TBM)

# Exact theorem: a family-universal full-rank proper vertex A=a I has polar factor exp(i arg a) I.
a=1.7-0.4j
Uuni=polar(a*np.eye(3,dtype=complex))
universal_offdiag=float(np.linalg.norm(np.abs(Uuni)-np.eye(3)))

obj={
 'status':'DEFINED_OBSERVABLE_REQUIRES_INTERACTING_VERTEX_PAYLOAD',
 'claim_status':'EXACT_EXTRACTION_MAP_PROVEN_NUMERICAL_PHYSICAL_CKM_PMNS_NOT_YET_DERIVED',
 'physical_payload_present':DATA.exists(),
 'definition':'V_phys = polar(Z_u^(1/2) Gamma_W_ud Z_d^(1/2)); U_phys = polar(Z_e^(1/2) Gamma_W_enu Z_nu^(1/2)) in one fixed renormalization/process convention.',
 'family_universal_vertex_theorem':{'offdiagonal_modulus_residual':universal_offdiag,'conclusion':'Family-universal gauge transport alone yields no physical mixing.'},
 'finite_TBM_reference_only':finite_reference,
 'finite_reference_scope':'This is the C3/TBM ultraviolet finite representative, not a physical PMNS prediction.',
 'required_payload':'data/g3_renormalized_weak_vertices.json',
 'external_mixing_data_used':False,
}
if DATA.exists():
    raw=json.loads(DATA.read_text()); obj['computed']={}
    for sec in ('quark','lepton'):
        A=np.array([[complex(*z) for z in row] for row in raw[sec]['vertex']],complex)
        U=polar(A); obj['computed'][sec]={'invariants':invs(U),'matrix':[[[float(z.real),float(z.imag)] for z in row] for row in U]}
    obj['status']='COMPUTED_FROM_INTERACTING_VERTEX_PAYLOAD'
(OUT/'MIXING_CLOSURE_CERTIFICATE.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
