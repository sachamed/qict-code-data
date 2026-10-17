#!/usr/bin/env python3
"""Quantify the flavor correction required beyond the finite receiver transfer.

External CKM/PMNS central values enter only in the post-comparison block.  The
receiver matrices themselves are read from the QICT-generated payload.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.linalg import logm
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'data'/'qict_receiver_mixing_exact.json'
OUT=ROOT/'results'/'global_sm_extension'/'FLAVOR_VERTEX_DEFICIT_QUANTITATIVE.json'
OUT.parent.mkdir(parents=True,exist_ok=True)

def cmatrix(rows):
    return np.array([[complex(float(z[0]),float(z[1])) for z in row] for row in rows],dtype=complex)

def standard_matrix(s12:float,s13:float,s23:float,delta:float)->np.ndarray:
    c12=math.sqrt(1-s12*s12); c13=math.sqrt(1-s13*s13); c23=math.sqrt(1-s23*s23)
    ep=complex(math.cos(delta),math.sin(delta)); em=ep.conjugate()
    return np.array([
      [c12*c13, s12*c13, s13*em],
      [-s12*c23-c12*s23*s13*ep, c12*c23-s12*s23*s13*ep, s23*c13],
      [s12*s23-c12*c23*s13*ep, -c12*s23-s12*c23*s13*ep, c23*c13]
    ],dtype=complex)

def inv(V:np.ndarray):
    s13=abs(V[0,2]); c13=math.sqrt(max(0.0,1-s13*s13))
    s12=abs(V[0,1])/c13; s23=abs(V[1,2])/c13
    J=float(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0])))
    return {
      'theta12_deg':math.degrees(math.asin(s12)),
      'theta13_deg':math.degrees(math.asin(s13)),
      'theta23_deg':math.degrees(math.asin(s23)),
      'J':J,
      'B_abs2':(np.abs(V)**2).tolist(),
    }

def best_rephasing(R:np.ndarray,T:np.ndarray):
    # D_L T D_R, with one redundant column phase fixed to zero.
    def objective(x):
        a=x[:3]; b=np.r_[x[3:5],0.0]
        A=np.diag(np.exp(1j*a))@T@np.diag(np.exp(1j*b))
        return float(np.linalg.norm(A-R,'fro')**2)
    rng=np.random.default_rng(20260917)
    best=None
    for n in range(64):
        x0=np.zeros(5) if n==0 else rng.uniform(-math.pi,math.pi,5)
        r=minimize(objective,x0,method='BFGS',options={'maxiter':3000,'gtol':1e-12})
        if best is None or r.fun<best.fun: best=r
    a=best.x[:3]; b=np.r_[best.x[3:5],0.0]
    A=np.diag(np.exp(1j*a))@T@np.diag(np.exp(1j*b))
    G=A@R.conj().T
    u,_,vh=np.linalg.svd(G); G=u@vh
    H=1j*logm(G); H=(H+H.conj().T)/2
    return A,G,H,a,b

def encode(M):
    return [[[float(z.real),float(z.imag)] for z in row] for row in M]

q=json.loads(SRC.read_text())
Rud=cmatrix(q['quark_receiver']['V']); Ren=cmatrix(q['lepton_receiver']['V'])
# External comparison values only.
Vckm=standard_matrix(0.22517,0.003763,0.04189,1.154) # PDG 2026 global fit
Vpmns=standard_matrix(math.sqrt(0.3088),math.sqrt(0.02248),math.sqrt(0.470),math.radians(212.0)) # NuFIT 6.1, NO, IC24+SK

def sector(label,R,T,source):
    ri,ti=inv(R),inv(T)
    B_R=np.abs(R)**2; B_T=np.abs(T)**2
    mod_lower=float(np.linalg.norm(np.abs(R)-np.abs(T),'fro'))
    A,G,H,a,b=best_rephasing(R,T)
    ev=np.linalg.eigvalsh(H)
    return {
      'receiver':ri,
      'external_target':ti,
      'external_source':source,
      'exact_rephasing_invariant_tests':{
        'frobenius_abs2_difference':float(np.linalg.norm(B_R-B_T,'fro')),
        'max_abs2_entry_difference':float(np.max(np.abs(B_R-B_T))),
        'frobenius_modulus_difference_lower_bound_on_any_rephased_matrix_distance':mod_lower,
        'J_difference':float(ri['J']-ti['J']),
      },
      'deterministic_multistart_rephasing_alignment':{
        'frobenius_distance':float(np.linalg.norm(A-R,'fro')),
        'spectral_distance':float(np.linalg.norm(A-R,2)),
        'unitary_correction_geodesic_frobenius':float(np.linalg.norm(H,'fro')),
        'unitary_correction_geodesic_spectral':float(np.linalg.norm(H,2)),
        'generator_eigenphases_rad':[float(x) for x in ev],
        'aligned_target':encode(A),
        'left_polar_correction_G':encode(G),
        'principal_hermitian_generator_H':encode(H),
        'row_phases_rad':[float(x) for x in a],
        'column_phases_rad':[float(x) for x in b],
      },
      'interpretation':'The exact modulus/J tests show that the finite receiver unitary is not the physical matrix. The polar correction quantifies the family-sensitive proper-vertex action that must be generated dynamically; it is not inserted into QICT as an input.'
    }

obj={
 'status':'EXECUTED_RECEIVER_MIXING_AND_VERTEX_DEFICIT',
 'external_values_used_only_after_receiver_matrices_were_generated':True,
 'receiver_source_sha256':__import__('hashlib').sha256(SRC.read_bytes()).hexdigest(),
 'CKM':sector('CKM',Rud,Vckm,'Particle Data Group 2026 CKM global fit, Eqs. (12.27)-(12.28)'),
 'PMNS':sector('PMNS',Ren,Vpmns,'NuFIT 6.1 (2025), normal ordering, IC24 with Super-K atmospheric data'),
 'theorem':'For any receiver unitary R and any target unitary P, a family-sensitive polar correction G=P R^dagger exists. Therefore R alone does not identify physical mixing unless the microscopic proper vertex fixes G. The nonzero rephasing-invariant modulus differences prove that G cannot be a family-universal phase correction for the present receiver matrices.'
}
OUT.write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
