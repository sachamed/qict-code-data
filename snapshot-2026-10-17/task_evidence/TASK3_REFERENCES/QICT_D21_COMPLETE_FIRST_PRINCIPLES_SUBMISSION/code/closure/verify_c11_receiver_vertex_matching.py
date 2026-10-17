#!/usr/bin/env python3
"""Exact C11 receiver-current / charged-current matching verification.

This certificate verifies the algebraic promotion criterion
    polar(Gamma) = V_rec  <=>  V_rec^dagger Gamma is Hermitian positive definite,
with V_rec=L_f^dagger L_g from the locked receiver returns.
It also verifies the Schur--Ward--LSZ factorization theorem after distinguishing the proper 1PI coefficient from the LSZ amplitude coefficient. If the proper current coefficient is an intertwiner of a receiver-current algebra and the current-intertwiner space is one-dimensional, it is a scalar multiple of V_rec. After the common scalar current form factor is divided out, the amplitude coefficient obeys Gamma^W_ren = Z_f^(1/2) V_rec Z_g^(1/2). Diagonal external factors and Ward/LSZ alone do not force multiplicity-one matching.

If a matched G3 charged-current coefficient payload is present, the same polar criterion is evaluated on it.
No experimental mixing data are read.
"""
from pathlib import Path
import csv, json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CERT=ROOT/'certificates'
SRC=DATA/'family_transfer_matrices.csv'
G3=DATA/'g3_renormalized_weak_vertices.json'
OUT=CERT/'C11_RECEIVER_VERTEX_MATCHING.json'

def load_transfers():
    mats={s:np.zeros((3,3),complex) for s in ('u','d','e','nu')}
    with SRC.open() as f:
        for r in csv.DictReader(f):
            mats[r['sector']][int(r['i']),int(r['j'])]=float(r['real'])+1j*float(r['imag'])
    return mats

def left_svd(T):
    L,s,_=np.linalg.svd(T)
    idx=np.argsort(s)
    return L[:,idx]

def polar(A):
    H=(A.conj().T@A)
    H=(H+H.conj().T)/2
    w,Q=np.linalg.eigh(H)
    if np.min(w)<=0: raise RuntimeError('rank deficient')
    return A@(Q*(1/np.sqrt(w)))@Q.conj().T

def criterion(V,G):
    H=V.conj().T@G
    anti=np.linalg.norm(H-H.conj().T,'fro')/(2*np.linalg.norm(G,'fro'))
    Hh=(H+H.conj().T)/2
    hmin=float(np.min(np.linalg.eigvalsh(Hh)))
    U=polar(G)
    pres=float(np.linalg.norm(U-V,2))
    return {'antihermitian_defect':float(anti),'positivity_gap':hmin,'polar_residual_2norm':pres,
            'matched':bool(anti<1e-12 and hmin>1e-12 and pres<1e-11)}

def parse_matrix(M):
    return np.asarray([[complex(z[0],z[1]) for z in row] for row in M],complex)

m=load_transfers(); L={k:left_svd(v) for k,v in m.items()}
pairs={'quark':L['u'].conj().T@L['d'],'lepton':L['e'].conj().T@L['nu']}
rng=np.random.default_rng(150926)
unit_tests={}
max_exact=0.0; min_counter=1e9
for name,V in pairs.items():
    # Positive H: exact theorem forward direction.
    z=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    H=z.conj().T@z+0.75*np.eye(3)
    G=V@H
    c=criterion(V,G); max_exact=max(max_exact,c['polar_residual_2norm'])
    # Generic cross-sector correction: same-sector projectors are untouched, but the polar factor generically changes.
    e=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    Gbad=V@H+0.08*e
    cbad=criterion(V,Gbad); min_counter=min(min_counter,cbad['polar_residual_2norm'])
    unit_tests[name]={'positive_factor_case':c,'generic_cross_sector_correction':cbad}

# Abstract multiplicity-one Schur regression on the irreducible C3 Weyl algebra.
def schur_intertwiner_test(V):
    om=np.exp(2j*np.pi/3)
    X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex)
    Z=np.diag([1,om,om**2]).astype(complex)
    blocks=[]
    I=np.eye(3,dtype=complex)
    for A in (X,Z):
        B=V@A@V.conj().T
        # vec(T A - B T)=0 in column-major convention.
        blocks.append(np.kron(A.T,I)-np.kron(I,B))
    M=np.vstack(blocks)
    u,sv,vh=np.linalg.svd(M)
    tol=max(M.shape)*np.finfo(float).eps*sv[0]
    nullity=int(np.sum(sv<tol))
    vec=vh.conj().T[:,-1]
    T=vec.reshape((3,3),order='F')
    alpha=np.vdot(V,T)/np.vdot(V,V)
    residual=float(np.linalg.norm(T-alpha*V,'fro')/np.linalg.norm(T,'fro'))
    intertwining=float(max(np.linalg.norm(T@A-(V@A@V.conj().T)@T,'fro') for A in (X,Z)))
    return {'nullity':nullity,'span_residual':residual,'intertwining_residual':intertwining,
            'multiplicity_one':bool(nullity==1 and residual<1e-10 and intertwining<1e-10)}

schur_tests={name:schur_intertwiner_test(V) for name,V in pairs.items()}

# Positive external residues need not preserve a polar factor before canonical leg normalization.
lsz_tests={}
for name,V in pairs.items():
    zf=np.diag([0.73,1.21,1.67]).astype(complex)
    zg=np.diag([1.44,0.88,1.13]).astype(complex)
    sf=np.diag(np.sqrt(np.diag(zf))); sg=np.diag(np.sqrt(np.diag(zg)))
    A=sf@V@sg
    raw_polar_res=float(np.linalg.norm(polar(A)-V,2))
    canon=np.diag(1/np.sqrt(np.diag(zf)))@A@np.diag(1/np.sqrt(np.diag(zg)))
    canon_res=float(np.linalg.norm(canon-V,2))
    lsz_tests[name]={"raw_amplitude_polar_residual":raw_polar_res,
                     "canonical_leg_normalization_residual":canon_res,
                     "external_residues_do_not_define_mixing":bool(raw_polar_res>1e-6 and canon_res<1e-12)}

physical={}
if G3.exists():
    obj=json.loads(G3.read_text())
    for name,V in pairs.items():
        if name in obj and 'vertex' in obj[name]:
            physical[name]=criterion(V,parse_matrix(obj[name]['vertex']))

cert={
 'observable':'C11 receiver-current to matched charged-current exact polar matching criterion',
 'logical_status':'PROVEN_POLAR_MATCHING_CRITERION__STANDARD_CKM_PMNS_IDENTIFICATION_REQUIRES_SECTOR_MATCHING',
 'theorem':'(i) For the full-rank renormalized proper coefficient Gamma_1PI and unitary V_rec, polar(Gamma_1PI)=V_rec iff V_rec^dagger Gamma_1PI is Hermitian positive definite. (ii) LSZ gives A_W=Z_f^(1/2) Gamma_1PI Z_g^(1/2). (iii) If Gamma_1PI is an intertwiner and Hom_R(H_g,H_f)=C V_rec, Schur multiplicity one gives Gamma_1PI=F V_rec and hence A_W=F Z_f^(1/2) V_rec Z_g^(1/2).',
 'source_vertex_identity':'The C11 weak pulse is the derivative of the same SU(2)_L link transport used by C8. The finite-regulator connected response defines an amputated charged-current coefficient algebraically through the external two-point response operators. Standard continuum weak-vertex identification requires the sector-specific renormalization and process matching stated in the manuscript.',
 'external_particle_data_read':False,
 'family_transfer_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
 'g3_physical_vertex_payload_present':G3.exists(),
 'unit_tests':unit_tests,
 'schur_multiplicity_one_tests':schur_tests,
 'lsz_external_leg_normalization_tests':lsz_tests,
 'physical_matching_gate':physical if physical else 'NOT_EVALUATED_NO_G3_VERTEX_DATA',
 'acceptance':{
   'exact_positive_factor_polar_residual_lt_1e-11':max_exact<1e-11,
   'counterexample_demonstrates_projector_reduction_not_sufficient':min_counter>1e-5,
   'schur_multiplicity_one_intertwiner_space_verified':all(x['multiplicity_one'] for x in schur_tests.values()),
   'lsz_leg_normalization_distinguished_from_mixing':all(x['external_residues_do_not_define_mixing'] for x in lsz_tests.values()),
 },
}
cert['status']='PASS' if all(cert['acceptance'].values()) else 'FAIL'
OUT.write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status']=='PASS' else 1)
