#!/usr/bin/env python3
r"""C11 family-current transition and interference observables.

The family-blind charged-current pulse acts as J0 \otimes I_3 on the family carrier.
When prepared and read out in the canonical non-degenerate left singular channels of
receiver returns T_f and T_g, the inclusive channel fraction is

    B_fg[a,b] = Tr(P^L_f,a P^L_g,b) = |(L_f^dagger L_g)[a,b]|^2.

Because the projectors reduce the complete compact C8J--C8K evolution, subsequent
family-blind gauge evolution cannot redistribute this inclusive channel weight.
The coherent four-amplitude interferometer probes the rephasing-invariant Bargmann
quartet.  These are operational C11 current observables.  They are not identified
with on-shell CKM/PMNS observables unless the separate proper-vertex matching gate
is established.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CERT=ROOT/'certificates'
SRC=DATA/'family_transfer_matrices.csv'
OUTCSV=DATA/'c11_family_current_observables.csv'
OUTCERT=CERT/'C11_FAMILY_CURRENT_OBSERVABLE.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_transfers():
    mats={s:np.zeros((3,3),complex) for s in ('u','d','e','nu')}
    with SRC.open() as f:
        for r in csv.DictReader(f):
            mats[r['sector']][int(r['i']),int(r['j'])]=float(r['real'])+1j*float(r['imag'])
    return mats


def left_svd(T):
    L,s,Vh=np.linalg.svd(T)
    idx=np.argsort(s)
    return L[:,idx],s[idx]


def quartets(V):
    vals=[]
    for a in range(3):
        for c in range(a+1,3):
            for b in range(3):
                for d in range(b+1,3):
                    q=V[a,b]*V[c,d]*np.conj(V[a,d])*np.conj(V[c,b])
                    vals.append((a,c,b,d,q))
    return vals

mats=load_transfers()
Ls={k:left_svd(v)[0] for k,v in mats.items()}
sv={k:left_svd(v)[1] for k,v in mats.items()}
pairs=[('u','d','quark'),('e','nu','lepton')]
rng=np.random.default_rng(210915)
rows=[]; results={}
max_rephase=0.0; max_common_basis=0.0; max_quartet_spread=0.0
for f,g,label in pairs:
    V=Ls[f].conj().T@Ls[g]
    B=np.abs(V)**2
    Q=quartets(V)
    jim=np.array([abs(float(np.imag(q[-1]))) for q in Q])
    J=float(jim.mean())
    spread=float(np.max(np.abs(jim-J)))
    max_quartet_spread=max(max_quartet_spread,spread)
    mean_change=float(1.0-np.trace(B).real/3.0)
    # Coherent quarter-turn contrast magnitude |I(-pi/2)-I(+pi/2)|=4|Im Q|.
    quarter_turn_contrast=float(4.0*J)
    # Numerical invariance regressions under arbitrary external rephasings and
    # arbitrary common unitary rotations of the family basis.
    for _ in range(32):
        pr=np.exp(1j*rng.uniform(-np.pi,np.pi,3))
        pc=np.exp(1j*rng.uniform(-np.pi,np.pi,3))
        Vr=np.diag(pr)@V@np.diag(pc)
        Br=np.abs(Vr)**2
        Jr=float(np.mean([abs(np.imag(q[-1])) for q in quartets(Vr)]))
        max_rephase=max(max_rephase,float(np.max(np.abs(Br-B))),abs(Jr-J))
        z=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
        W,r=np.linalg.qr(z)
        ph=np.diag(r); ph=np.where(np.abs(ph)>0,ph/np.abs(ph),1.0)
        W=W@np.diag(np.conj(ph))
        Vb=(W@Ls[f]).conj().T@(W@Ls[g])
        max_common_basis=max(max_common_basis,float(np.linalg.norm(Vb-V,ord=2)))
    for a in range(3):
        for b in range(3):
            rows.append({'pair':label,'observable':'inclusive_transition_fraction','from_channel':a+1,'to_channel':b+1,'value':float(B[a,b])})
    rows.append({'pair':label,'observable':'equal_source_mean_channel_change','from_channel':'all','to_channel':'off_diagonal','value':mean_change})
    rows.append({'pair':label,'observable':'abs_C11_Bargmann_J','from_channel':'quartet','to_channel':'quartet','value':J})
    rows.append({'pair':label,'observable':'quarter_turn_interference_contrast','from_channel':'coherent_pair','to_channel':'coherent_pair','value':quarter_turn_contrast})
    results[label]={
        'from_sector':f,'to_sector':g,
        'channel_basis':'ascending non-degenerate singular channels of the fixed receiver returns',
        'inclusive_transition_fraction_matrix':B.tolist(),
        'equal_source_mean_channel_change_fraction':mean_change,
        'abs_C11_Bargmann_J':J,
        'quarter_turn_interference_contrast_abs':quarter_turn_contrast,
        'quartet_abs_imag_spread':spread,
        'unitarity_residual_2norm':float(np.linalg.norm(V.conj().T@V-np.eye(3),2)),
        'row_stochastic_residual_inf':float(np.max(np.abs(B.sum(axis=1)-1))),
        'column_stochastic_residual_inf':float(np.max(np.abs(B.sum(axis=0)-1))),
    }

with OUTCSV.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['pair','observable','from_channel','to_channel','value']); w.writeheader(); w.writerows(rows)

min_gap=min(float(np.min(np.diff(s))) for s in sv.values())
cert={
    'observable':'C11 inclusive family-current transition fractions and coherent Bargmann interference invariant',
    'logical_status':'EXACT_C11_CURRENT_OBSERVABLE_WITH_DECLARED_RECEIVER_CLOSURE',
    'operational_definition':(
        'Prepare a normalized canonical left singular channel in sector f, apply a family-universal charged-current pulse J0 tensor I3, '
        'and resolve the outgoing canonical left singular channel in sector g while summing all non-family degrees of freedom. The normalized '
        'inclusive fraction is B_fg[a,b]. For the coherent phase test, interfere the two composite amplitudes V_ab V_cd and V_ad V_cb; the '
        'absolute quarter-turn contrast is 4 |Im(V_ab V_cd V_ad* V_cb*)|.'
    ),
    'scope_guard':(
        'These quantities are directly operational within the C11 receiver-current protocol because the singular projectors reduce the full '
        'family-blind compact C8J-C8K evolution. They are not called CKM/PMNS observables and are not substituted for the on-shell renormalized '
        'proper-vertex polar factors required by G3.'
    ),
    'closure_dependencies':['A1-A8','LP1 complementary-arc route prescription','LP2 minimal-flux representation-cost prescription'],
    'external_particle_data_read':False,
    'family_transfer_source_sha256':sha256(SRC),
    'minimum_singular_value_gap':min_gap,
    'pairs':results,
    'invariance_regression':{
        'random_rephasing_trials_per_pair':32,
        'maximum_B_or_J_rephasing_residual':max_rephase,
        'maximum_common_family_basis_rotation_residual_2norm':max_common_basis,
        'maximum_unitary_quartet_abs_imag_spread':max_quartet_spread,
    },
}
accept={
    'simple_singular_channels':min_gap>1e-6,
    'unitarity_lt_2e-12':all(v['unitarity_residual_2norm']<2e-12 for v in results.values()),
    'stochasticity_lt_2e-12':all(max(v['row_stochastic_residual_inf'],v['column_stochastic_residual_inf'])<2e-12 for v in results.values()),
    'quartet_equality_lt_1e-15':max_quartet_spread<1e-15,
    'rephasing_invariance_lt_1e-12':max_rephase<1e-12,
    'common_basis_invariance_lt_1e-12':max_common_basis<1e-12,
}
cert['acceptance']=accept; cert['status']='PASS' if all(accept.values()) else 'FAIL'
OUTCERT.write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status']=='PASS' else 1)
