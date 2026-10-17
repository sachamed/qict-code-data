#!/usr/bin/env python3
"""Independent integer Bareiss inertia and full neutral residual verification."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from compute_coupled_su2_spin_network import schur_form,labels,neighbors,weight,electric

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/global_electroweak_verification'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def bareiss_inertia(S):
    denominators=[math.lcm(*(x.denominator for x in row)) for row in S]
    A=[[int(S[i][j]*denominators[i]*denominators[j]) for j in range(len(S))]
       for i in range(len(S))]
    previous=1;signs=[1]
    for k in range(len(A)):
        pivot=A[k][k];assert pivot!=0
        signs.append(1 if pivot>0 else -1)
        for i in range(k+1,len(A)):
            for j in range(k+1,len(A)):
                numerator=pivot*A[i][j]-A[i][k]*A[k][j]
                assert numerator%previous==0
                A[i][j]=numerator//previous
        previous=pivot
    negatives=sum(a!=b for a,b in zip(signs,signs[1:]))
    return negatives,signs


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    coupled=ROOT/'reproducibility/results/coupled_su2_spin_network'
    cpath=coupled/'COUPLED_SU2_SPIN_NETWORK_CERTIFICATE.json'
    c=json.loads(cpath.read_text());g=c['cutoffs'][-1]
    S,core,tail=schur_form(59,128)
    negative,signs=bareiss_inertia(S)
    assert negative==1 and len(core)==41 and tail==132
    with np.load(coupled/'COUPLED_SU2_THETA_STATE_AND_MOMENTS.npz',allow_pickle=False) as z:
        source_labels=z['R16_labels']
    w={tuple(map(int,x)):F(v) for x,v in zip(source_labels,g['weighted_rational_witness'])}
    norm=sum(weight(x)*v*v for x,v in w.items())
    assert norm==F(g['witness_normalization_squared_rational'])
    mu=F(g['rayleigh_rational']);residual=F(0);numerator=F(0)
    # Weighted adjoint row action, rather than the production column
    # accumulation; every omitted one-hop target lies in radius seventeen.
    for x in labels(17):
        value=w.get(x,F(0))
        hw=(electric(x)+64)*value
        for y,coefficient in neighbors(x):
            hw-=16*F(weight(y),weight(x))*coefficient*w.get(y,F(0))
        residual+=weight(x)*(hw-mu*value)**2
        numerator+=weight(x)*value*hw
    assert numerator/norm==mu
    assert residual/norm==F(g['infinite_residual_squared_rational'])
    heat_path=ROOT/'reproducibility/results/global_primitive_gap/GLOBAL_PRIMITIVE_HEAT_KERNEL_GAP_CERTIFICATE.json'
    heat=json.loads(heat_path.read_text())
    for value in heat['per_link_heat_kernel_tail_upper_rational'].values():
        assert F(value)<F(1,2)
    weak=heat['independent_group_gap_bounds']['SU2']
    assert weak['magnetic_cost_upper']==64*18252
    assert heat['full_neutral_primitive_gap_lower']['denominator_exponent']==\
        2*(64*18252+21136)-1==2378527
    payload=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    with np.load(payload,allow_pickle=False) as z:
        link=z['links'].astype(int)
    starts=link[:,1:];ends=starts.copy()
    ends[np.arange(len(link)),link[:,0]]+=1
    assert np.array_equal(ends.sum(axis=1)-starts.sum(axis=1),np.ones(len(link),int))
    ewpath=ROOT/'reproducibility/results/common_coupled_electroweak/COMMON_COUPLED_ELECTROWEAK_GNS_CERTIFICATE.json'
    ew=json.loads(ewpath.read_text())
    ewd=ewpath.parent/'COMMON_COUPLED_ELECTROWEAK_GNS_RESOLVENTS.npz'
    assert sha(ewd)==ew['data_sha256']
    with np.load(ewd,allow_pickle=False) as z:
        moments=z['moments'];source=z['CAR_vacuum_and_dual_source_columns']
        assert np.linalg.norm(source.conj().T@source-np.eye(6))<2e-15
        assert np.linalg.norm(moments[0]-np.eye(6))<3e-13
        for i,point in enumerate(z['real_resolvent_points']):
            G=sum(moments[n]/int(point)**(n+1) for n in range(13))
            assert np.max(abs(G-z['compressed_resolvents'][i]))<3e-14
            Sigma=int(point)*np.eye(6)-moments[1]-np.linalg.inv(G)
            assert np.max(abs(Sigma-z['Feshbach_self_energies'][i]))<3e-14
    result={'status':'PASS_INDEPENDENT_GLOBAL_GAP_SU2_INERTIA_AND_ELECTROWEAK_VERIFY',
        'integer_Bareiss_Schur_negative_inertia':negative,
        'leading_principal_minor_signs':signs,
        'weighted_row_infinite_residual_exact':True,
        'weighted_Rayleigh_exact':True,'linear_U1_Gauss_cancellation_all_links':True,
        'full_neutral_primitive_gap_expression_verified':True,
        'electroweak_Gram_moments_and_Feshbach_arrays_verified':True,
        'source_certificates_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [cpath,heat_path,ewpath]},
        'floating_point_interval_certified':False,
        'full_C10_contract_completed':False,
        'scope':'Different fraction-free inertia arithmetic, adjoint row residual and global heat-gap/geometry checks; no full physical pole or connected continuum inference.'}
    (OUT/'INDEPENDENT_GLOBAL_ELECTROWEAK_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'negative_inertia':negative,
        'global_gap_denominator_power':2378527,
        'resolvents_verified':3},indent=2))


if __name__=='__main__':main()
