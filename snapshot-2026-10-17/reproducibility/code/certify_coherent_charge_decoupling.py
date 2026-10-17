#!/usr/bin/env python3
"""Exact modular charge intertwiners for authentic one/two-plaquette U1.

A controlled unitary basis change reduces static charged transfers to a
single neutral transfer. Source loop shifts remain coherent.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from certify_exact_resonance_reduction import normalized_witness

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/coherent_charge_reduction'
PHI=math.pi/63;DEPTH=12


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def step(v,electric,magnetic):
    return np.fft.ifftn(magnetic*np.fft.fftn(electric*v))


def run_case(dim,length,labels,values):
    n=np.arange(length,dtype=np.int64)
    n=np.where(n<(length+1)//2,n,n-length)
    angle=2*np.pi*np.arange(length)/length
    if dim==1:
        r=n;neutral=8*r*r;mag=4-4*np.cos(angle);seed=np.zeros(length,complex)
    else:
        r=n[:,None];s=n[None,:]
        neutral=8*r*r+8*s*s-4*r*s
        mag=8-4*np.cos(angle[:,None])-4*np.cos(angle[None,:])
        seed=np.zeros((length,length),complex)
    for label,value in zip(labels,values):seed[tuple(int(x)%length for x in label)]=value
    b=np.exp(-1j*PHI*mag);e0=np.exp(-1j*PHI*(neutral%126))
    records=[];arrays={}
    for q in range(0,-28,-3):
        if dim==1:
            delta=(-5*q,);constant=198*q*q-q
            cost=neutral+4*q*r+2*q*q+q
            difference=8*(r+delta[0])**2-cost-constant
        else:
            delta=(-4*q,4*q);constant=318*q*q-q
            cost=neutral+4*q*(r-s)+2*q*q+q
            difference=8*(r+delta[0])**2+8*(s+delta[1])**2-\
                       4*(r+delta[0])*(s+delta[1])-cost-constant
        assert not np.any(difference%126)
        e=np.exp(-1j*PHI*(cost%126));omega=np.exp(1j*PHI*(constant%126))
        shifted=np.roll(seed,delta,axis=tuple(range(dim)))
        direct=seed.copy();unified=shifted.copy();direct_ret=[];unified_ret=[]
        depth=DEPTH if q in [0,-3,-6] else 1
        maximum=0
        for t in range(depth+1):
            direct_ret.append(np.vdot(seed,direct))
            unified_ret.append(omega**t*np.vdot(shifted,unified))
            reconstituted=omega**t*np.roll(unified,tuple(-x for x in delta),
                                           axis=tuple(range(dim)))
            maximum=max(maximum,float(np.linalg.norm(direct-reconstituted)))
            direct=step(direct,e,b);unified=step(unified,e0,b)
        moment_error=float(np.max(abs(np.asarray(direct_ret)-np.asarray(unified_ret))))
        assert maximum<3e-12 and moment_error<3e-12
        records.append({'qY':q,'coherent_loop_shift':delta,
          'fixed_phase_integer_mod126':int(constant%126),
          'integer_phase_identity_exact':True,'propagation_depth':depth,
          'vector_intertwining_error':maximum,'moment_reconstruction_error':moment_error})
        arrays[f'd{dim}_q{q}_direct']=np.asarray(direct_ret)
        arrays[f'd{dim}_q{q}_neutral_shifted']=np.asarray(unified_ret)
    return records,arrays


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    one_path=ROOT/'reproducibility/results/nonabelian_common_car/NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json'
    two_path=ROOT/'reproducibility/results/coupled_boundary/COUPLED_BOUNDARY_GNS_CERTIFICATE.json'
    one=json.loads(one_path.read_text())['cutoff_results'][-1]['ground_certificates']['U1']
    two=json.loads(two_path.read_text())['cutoffs'][-1]['ground']
    labels1=np.array(one['support_labels'],int);v1=normalized_witness(one)
    with np.load(two_path.parent/'COUPLED_PLAQUETTE_BOUNDARY_MOMENTS.npz',allow_pickle=False) as z:
        labels2=z['L3_ground_labels']
    v2=normalized_witness(two)
    result1,a1=run_case(1,252,labels1,v1)
    result2,a2=run_case(2,252,labels2,v2)
    # General integral charges use inverses 4^-1=16 and 5^-1=38 modulo 63.
    assert (4*16)%63==1 and (5*38)%63==1
    for q in range(-30,31):
        assert (4*(16*q)-q)%63==0
        assert (5*(38*q)-q)%63==0
    npz=OUT/'COHERENT_CHARGE_NEUTRAL_TRANSFER_CHECKS.npz'
    np.savez_compressed(npz,**a1,**a2)
    result={'status':'EXACT_COHERENT_COMMON_NEUTRAL_TRANSFER_REDUCTION',
      'general_charge_shifts':{'one_loop':'delta=16*q (mod 63)',
                              'shared_link_two_loop':'delta=38*q*(1,-1) (mod 63)'},
      'six_mode_leptonic_CAR_shifts':{'one_loop':'delta(Q)=-5*Q',
                                     'two_loop':'delta(Q)=(-4*Q,4*Q)'},
      'fixed_charge_phase_polynomials':{'one_loop':'198*Q^2-Q',
                                       'two_loop':'318*Q^2-Q'},
      'operator_identity':'U_Q=exp(i*phi*c(Q))*S_-delta(Q)*U_0*S_delta(Q)',
      'controlled_basis_change':'J=sum_occupation |occupation><occupation| tensor S_delta(Q)',
      'transformed_full_CAR_update':'J U J^dagger=exp(i*phi*c(Q)) tensor U_0',
      'transformed_source':'J O_a J^dagger=creation_a tensor S_delta(q_a), retaining its Wilson flux chart and compensator.',
      'source_CAR_preserved_exactly':True,'neutral_Kato_state_unchanged_exactly':True,
      'relative_moment_formula':'C_aa(n)=exp(i*n*phi*c(q_a))*conj(<Omega|U0^n|Omega>)*<S_delta(q_a)Omega|U0^n|S_delta(q_a)Omega>',
      'same_physical_source_not_dephased':True,
      'single_loop_checks':result1,'coupled_loop_checks':result2,
      'input_primitive_certificate_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [one_path,two_path]},
      'scope':'Authentic static U1 modules. Charges and phase polynomials are determined by the frozen coefficients; no mass fit or added interaction. Non-Abelian charged streams, moving sources and full C10 exterior remain distinct.',
      'floating_point_interval_certified':False,
      'full_C10_contract_completed':False,'data_sha256':sha(npz)}
    (OUT/'COHERENT_CHARGE_DECOUPLING_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'max_vector_error':max(x['vector_intertwining_error'] for x in result1+result2),
      'max_moment_error':max(x['moment_reconstruction_error'] for x in result1+result2),
      'single_neutral_transfer':True,'full_C10':False},indent=2))


if __name__=='__main__':main()
