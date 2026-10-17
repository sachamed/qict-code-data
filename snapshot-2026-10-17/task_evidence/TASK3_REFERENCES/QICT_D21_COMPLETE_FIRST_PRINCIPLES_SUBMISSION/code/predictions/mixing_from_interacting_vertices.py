#!/usr/bin/env python3
"""Prospective parameter-free CKM/PMNS extraction from renormalized interacting weak vertices."""
from pathlib import Path
import json, math, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CERT=ROOT/'certificates'
INP=DATA/'g3_renormalized_weak_vertices.json'; LOCK=ROOT/'preregistration'/'QICT_PREDICTION_LOCK.json'
OUT=DATA/'qict_prospective_ckm_pmns.json'; COUT=CERT/'QICT_PROSPECTIVE_MIXING.json'
def polar(A):
    H=A.conj().T@A; w,V=np.linalg.eigh((H+H.conj().T)/2)
    if np.min(w)<=0: raise RuntimeError('proper vertex is rank deficient')
    return A@(V*(1/np.sqrt(w)))@V.conj().T
def invariants(U):
    s13=abs(U[0,2]); c13=math.sqrt(max(0,1-s13*s13))
    s12=abs(U[0,1])/c13; s23=abs(U[1,2])/c13
    c12=math.sqrt(max(0,1-s12*s12)); c23=math.sqrt(max(0,1-s23*s23))
    J=float(np.imag(U[0,0]*U[1,1]*np.conj(U[0,1])*np.conj(U[1,0])))
    den=max(1e-300,s12*c12*s23*c23*s13*c13*c13)
    sind=max(-1,min(1,J/den))
    den2=max(1e-300,2*s12*c12*c23*s23*s13)
    cosd=(abs(U[1,0])**2-s12**2*c23**2-c12**2*s23**2*s13**2)/den2
    cosd=max(-1,min(1,cosd)); delta=math.atan2(sind,cosd)
    if delta<0: delta+=2*math.pi
    return {'theta12_rad':math.asin(s12),'theta13_rad':math.asin(s13),'theta23_rad':math.asin(s23),'J':J,'delta_rad':delta,'unitarity_residual':float(np.linalg.norm(U.conj().T@U-np.eye(3),2))}
def parse(M):
    return np.asarray([[complex(z[0],z[1]) for z in row] for row in M],complex)
if not (INP.exists() and LOCK.exists()):
    out={'status':'DEFINED_OBSERVABLE','claim_status':'DEFINED_OBSERVABLE','reason':'Numerical evaluation requires the renormalized gauge-independent interacting weak-vertex payload and fixed precomparison specification. Finite holonomies are excluded as substitutes for physical CKM/PMNS matrices.','external_mixing_data_read':False}
    COUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); raise SystemExit(0)
obj=json.loads(INP.read_text()); out={'status':'COMPUTED','claim_status':'COMPUTED','external_mixing_data_read':False,'sectors':{}}
for sec in ('quark','lepton'):
    A=parse(obj[sec]['vertex']); U=polar(A); out['sectors'][sec]={'U':[[[float(z.real),float(z.imag)] for z in row] for row in U],**invariants(U)}
OUT.write_text(json.dumps(out,indent=2)+'\n'); COUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
