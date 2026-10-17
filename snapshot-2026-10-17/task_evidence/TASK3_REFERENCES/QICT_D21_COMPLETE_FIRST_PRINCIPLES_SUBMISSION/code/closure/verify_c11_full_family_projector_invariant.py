#!/usr/bin/env python3
"""Exact full-C11 family reducing-projector and channel-overlap invariant.

The C8J gauge stream and all gauge-field layers act trivially on the three-dimensional
family carrier.  For a non-degenerate receiver return T_f=L_f S_f R_f^dagger, the
singular projectors therefore reduce the full gauged Julia operator for every unitary
compact gauge stream A_q[u].  C8K preserves this direct-sum decomposition under the
CAR functor.  Consequently matrix-valued C11 moments have no off-diagonal leakage
between singular channels of a fixed incidence sector at any regulator.

The left-channel overlap B_fg[a,b]=Tr(P^L_f,a P^L_g,b)=|(L_f^dagger L_g)_ab|^2 is
then a basis- and rephasing-invariant family geometry carried unchanged through the
full C11 regulator net.  It is not, by itself, the renormalized physical CKM/PMNS
polar factor; that proper-vertex observable remains governed by the C11 promotion gate.
"""
from pathlib import Path
import csv, json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CERT=ROOT/'certificates'
SRC=DATA/'family_transfer_matrices.csv'
OUTCSV=DATA/'c11_full_family_projector_overlap.csv'
OUTCERT=CERT/'C11_FULL_FAMILY_PROJECTOR_INVARIANT.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_transfers():
    mats={s:np.zeros((3,3),complex) for s in ('u','d','e','nu')}
    with SRC.open() as f:
        for r in csv.DictReader(f):
            mats[r['sector']][int(r['i']),int(r['j'])]=float(r['real'])+1j*float(r['imag'])
    return mats

def svd_ascending(T):
    L,s,Vh=np.linalg.svd(T)
    idx=np.argsort(s)
    return L[:,idx],s[idx],Vh.conj().T[:,idx]

def haar_unitary(n,rng):
    z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    q,r=np.linalg.qr(z)
    d=np.diag(r)
    phase=np.where(np.abs(d)>0,d/np.abs(d),1.0)
    return q@np.diag(np.conj(phase))

def julia(T,A):
    L,s,R=svd_ascending(T)
    d=np.sqrt(np.maximum(0.0,1.0-s*s))
    DL=(L*d)@L.conj().T
    DR=(R*d)@R.conj().T
    n=A.shape[0]
    return np.block([[np.kron(A,DL), 1j*np.kron(np.eye(n),T)],
                     [1j*np.kron(np.eye(n),T.conj().T), np.kron(A.conj().T,DR)]])

def channel_projector(T,a,n):
    L,s,R=svd_ascending(T)
    pl=np.outer(L[:,a],L[:,a].conj())
    pr=np.outer(R[:,a],R[:,a].conj())
    z=np.zeros((3*n,3*n),complex)
    return np.block([[np.kron(np.eye(n),pl),z],[z,np.kron(np.eye(n),pr)]])


mats=load_transfers()
svd={s:svd_ascending(T) for s,T in mats.items()}
min_gap=min(float(np.min(np.diff(vals[1]))) for vals in svd.values())
max_intertwiner=0.0
max_projector_orth=0.0
for sec,T in mats.items():
    L,s,R=svd[sec]
    for a in range(3):
        pl=np.outer(L[:,a],L[:,a].conj()); pr=np.outer(R[:,a],R[:,a].conj())
        max_intertwiner=max(max_intertwiner,float(np.linalg.norm(pl@T-T@pr,ord='fro')))
        for b in range(3):
            if a!=b:
                plb=np.outer(L[:,b],L[:,b].conj())
                prb=np.outer(R[:,b],R[:,b].conj())
                max_projector_orth=max(max_projector_orth,float(np.linalg.norm(pl@plb,'fro')),float(np.linalg.norm(pr@prb,'fro')))

# Arbitrary compact gauge-stream test.  The commutator theorem is analytic; this is an
# independent floating-point regression across dimensions and random unitary streams.
rng=np.random.default_rng(210911)
max_comm=0.0
trials=[]
for n in (2,3,5,8):
    for rep in range(5):
        A=haar_unitary(n,rng)
        for sec,T in mats.items():
            J=julia(T,A)
            local=0.0
            for a in range(3):
                Q=channel_projector(T,a,n)
                local=max(local,float(np.linalg.norm(Q@J-J@Q,ord='fro')))
            max_comm=max(max_comm,local)
        trials.append({'stream_dimension':n,'trial':rep+1})

pairs=[('u','d','quark'),('e','nu','lepton')]
rows=[]; pair_results={}
for f,g,label in pairs:
    Lf=svd[f][0]; Lg=svd[g][0]
    V=Lf.conj().T@Lg
    B=np.abs(V)**2
    for a in range(3):
        for b in range(3):
            rows.append({'pair':label,'from_sector':f,'to_sector':g,'a_ascending':a+1,'b_ascending':b+1,'projector_overlap':float(B[a,b])})
    pair_results[label]={
        'from_sector':f,'to_sector':g,
        'left_overlap_moduli':np.abs(V).tolist(),
        'projector_overlap_matrix':B.tolist(),
        'unitarity_residual_2norm':float(np.linalg.norm(V.conj().T@V-np.eye(3),2)),
        'double_stochastic_row_residual_inf':float(np.max(np.abs(B.sum(axis=1)-1))),
        'double_stochastic_column_residual_inf':float(np.max(np.abs(B.sum(axis=0)-1))),
        'projector_mixing_purity_sum_B2':float(np.sum(B*B)),
        'abs_det_projector_overlap':float(abs(np.linalg.det(B))),
    }

with OUTCSV.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

cert={
    'observable':'full-C11 family reducing-projector and channel-overlap invariant',
    'logical_status':'EXACT_FULL_C11_CHANNEL_INVARIANT',
    'scope':(
        'For each incidence sector f, the non-degenerate singular projectors of T_f reduce the complete '
        'family-blind C8J gauge stream and its C8K CAR lift at every regulator. Hence off-diagonal C11 '
        'moments between distinct singular channels vanish identically. The cross-sector left-projector '
        'overlap is a regulator-independent family invariant. It is not identified with the renormalized '
        'physical CKM/PMNS polar factor unless the separate interacting proper-vertex factorization gate is proved.'
    ),
    'external_particle_data_read':False,
    'family_transfer_source_sha256':sha(SRC),
    'minimum_singular_value_gap':min_gap,
    'maximum_singular_intertwiner_residual_fro':max_intertwiner,
    'maximum_projector_orthogonality_residual_fro':max_projector_orth,
    'random_compact_gauge_stream_trials':len(trials),
    'maximum_full_gauged_julia_projector_commutator_residual_fro':max_comm,
    'exact_C11_same_sector_leakage_invariant':0.0,
    'regulator_convergence_statement':'identically zero leakage and constant projector overlaps at every finite regulator; Moore-Smith net is constant, so convergence error is exactly zero',
    'pairs':pair_results,
    'acceptance':{
        'min_singular_gap_gt_1e-6':min_gap>1e-6,
        'intertwiner_residual_lt_1e-12':max_intertwiner<1e-12,
        'projector_orthogonality_lt_1e-12':max_projector_orth<1e-12,
        'gauged_julia_commutator_lt_1e-11':max_comm<1e-11,
        'overlap_unitarity_lt_1e-12':all(x['unitarity_residual_2norm']<1e-12 for x in pair_results.values()),
        'double_stochastic_lt_1e-12':all(max(x['double_stochastic_row_residual_inf'],x['double_stochastic_column_residual_inf'])<1e-12 for x in pair_results.values()),
    }
}
cert['status']='PASS' if all(cert['acceptance'].values()) else 'FAIL'
OUTCERT.write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status']=='PASS' else 1)
