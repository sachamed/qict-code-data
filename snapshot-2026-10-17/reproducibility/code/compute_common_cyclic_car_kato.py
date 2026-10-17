#!/usr/bin/env python3
"""Common cyclic-flux regulator, genuine six-mode CAR, coupled Kato state.

Actual shared-link U1 component only. The cyclic boundary is a controlled
regulator convention, not a replacement of the continuous U1 gauge group.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from certify_exact_resonance_reduction import normalized_witness
from nonabelian_common_car_kato import polynomial_tail

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/common_cyclic_car'
PRIOR=ROOT/'reproducibility/results/coupled_boundary'
PAYLOAD=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
PHI=math.pi/63;DEPTH=12


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def car():
    creators=[]
    charges=[]
    q=np.array([-3]*3+[-6]*3)
    for a in range(6):
        matrix=np.zeros((64,64),np.int64)
        for n in range(64):
            if not ((n>>a)&1):
                matrix[n|(1<<a),n]=(-1)**((n&((1<<a)-1)).bit_count())
        creators.append(matrix)
    for a in range(6):
        for b in range(6):
            assert np.array_equal(creators[a].T@creators[b]+creators[b]@creators[a].T,
                                  np.eye(64,dtype=int) if a==b else np.zeros((64,64),int))
            assert np.count_nonzero(creators[a]@creators[b]+creators[b]@creators[a])==0
    for n in range(64):charges.append(int(sum(q[a]*((n>>a)&1) for a in range(6))))
    sources=np.column_stack([c[:,0] for c in creators])
    assert np.array_equal(sources.T@sources,np.eye(6,dtype=int))
    return charges


def update(v,pe,pb,inverse=False):
    if inverse:return pe.conj()*np.fft.ifft2(pb.conj()*np.fft.fft2(v))
    return np.fft.ifft2(pb*np.fft.fft2(pe*v))


def returns(seed,pe,pb):
    v=seed.copy();back=seed.copy();pos=[];neg=[];norm=0
    for n in range(DEPTH+1):
        pos.append(np.vdot(seed,v));neg.append(np.vdot(seed,back))
        norm=max(norm,float(abs(np.vdot(v,v)-1)))
        v=update(v,pe,pb);back=update(back,pe,pb,True)
    adjoint=float(np.max(abs(np.asarray(neg)-np.asarray(pos).conj())))
    assert adjoint<2e-12
    return np.asarray(pos),norm,adjoint


def gauss_check(data):
    with np.load(PAYLOAD,allow_pickle=False) as z:
        links=z['links'][data['actual_link_indices']].astype(int)
    starts=links[:,1:];ends=starts.copy();ends[np.arange(7),links[:,0]]+=1
    vertices=sorted(set(map(tuple,starts))|set(map(tuple,ends)))
    index={x:i for i,x in enumerate(vertices)};boundary=np.zeros((len(vertices),7),int)
    for e,(a,b) in enumerate(zip(starts,ends)):
        boundary[index[tuple(a)],e]-=1;boundary[index[tuple(b)],e]+=1
    B=data['integer_cycle_matrix'].astype(int)
    assert not np.any(boundary@B)
    shared=int(np.flatnonzero(data['actual_link_indices']==int(data['source_positive_edge']))[0])
    h=np.eye(7,dtype=int)[:,shared]
    constraint=boundary@h
    assert np.count_nonzero(constraint)==2
    return {'vertices':len(vertices),'links':7,'cycle_Gram':(B.T@B).tolist(),
      'neutral_integer_Gauss_residual':0,
      'Wilson_dressing_field_boundary':constraint.tolist(),
      'source_and_dual_endpoint_cancel_field_Gauss_exactly':True}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old_cert=PRIOR/'COUPLED_BOUNDARY_GNS_CERTIFICATE.json'
    old_data=PRIOR/'COUPLED_PLAQUETTE_BOUNDARY_MOMENTS.npz'
    certificate=json.loads(old_cert.read_text())
    ground=certificate['cutoffs'][-1]['ground']
    values=normalized_witness(ground)
    with np.load(old_data,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    labels=data['L3_ground_labels'];assert len(values)==len(labels)
    mu=F(ground['rayleigh_rational']);D=F(ground['state_trace_distance_upper_rational'])
    gap=8-mu
    assert gap>0
    # For q=-3, u=r+s and t=r-s have equal parity:
    # KE=3*u^2+5*(t-6/5)^2+39/5, and the exact integer minimum is 11.
    candidates=[3*u*u+5*t*t-12*t+15 for u in range(-8,9)
                for t in range(-8,9) if (u-t)%2==0]
    assert min(candidates)==11
    assert 3*9**2+F(39,5)>11
    assert 5*(F(9)-F(6,5))**2+F(39,5)>11
    # All other nonzero Fock charges are <=-6; the continuous lower
    # minimum 6*q^2/5+q is >=186/5>8.
    charges=car();assert set(charges)==set(range(-27,1,3))
    assert min(F(6*q*q,5)+q for q in set(charges) if q<=-6)==F(186,5)
    arrays={};records=[];previous=None
    for samples in [1,2,4,8]:
        length=63*samples
        k=np.arange(length,dtype=np.int64);k=np.where(k<(length+1)//2,k,k-length)
        r=k[:,None];s=k[None,:]
        ke=8*r*r+8*s*s-4*r*s
        angle=2*np.pi*np.arange(length)/length
        kb=8-4*np.cos(angle[:,None])-4*np.cos(angle[None,:])
        pb=np.exp(-1j*PHI*kb)
        seed=np.zeros((length,length),complex)
        for (a,b),value in zip(labels,values):seed[int(a)%length,int(b)%length]=value
        factors=[];errors=[]
        for q in [0,-3,-6]:
            pe=np.exp(-1j*PHI*((ke+4*q*(r-s)+2*q*q+q)%126))
            a,norm,adjoint=returns(seed,pe,pb)
            factors.append(a);errors.append({'qY':q,'norm_error':norm,'adjoint_error':adjoint})
        a=np.asarray(factors)
        c=np.array([np.diag([a[0,n].conj()*a[1,n]]*3+
                            [a[0,n].conj()*a[2,n]]*3) for n in range(DEPTH+1)])
        toeplitz=np.block([[c[j-i] if j>=i else c[i-j].conj().T
                           for j in range(DEPTH+1)] for i in range(DEPTH+1)])
        mineig=float(np.linalg.eigvalsh(toeplitz).min())
        assert mineig>-1e-11 and np.linalg.norm(c[0]-np.eye(6))<1e-12
        order=(length-2*int(np.max(abs(labels)))-1)//DEPTH
        tail=polynomial_tail(F(2,5),order)
        alias=4*DEPTH*tail*(1+tail)**(DEPTH-1)
        bound=4*D+alias
        difference=float(np.max(abs(c-data['L3_Cn6'])))
        assert difference<3e-12
        rec={'cyclic_length_per_axis':length,'gauge_dimension':length*length,
          'full_CAR_carrier_dimension':64*length*length,
          'vacuum_plus_six_source_carrier_dimension':7*length*length,
          'common_neutral_and_charged_regulator':True,
          'uniform_homotopy_Kato_gap_lower_rational':str(gap),
          'uniform_homotopy_Kato_gap_lower':float(gap),
          'finite_regulator_state_error_in_moment_rational':str(4*D),
          'infinite_flux_state_and_alias_moment_bound_rational':str(bound),
          'infinite_flux_state_and_alias_moment_bound_upper':min(2.,float(bound)),
          'independent_buffered_moment_difference':difference,
          'moment_successive_change':None if previous is None else float(np.max(abs(c-previous))),
          'C0_Frobenius_error':float(np.linalg.norm(c[0]-np.eye(6))),
          'Toeplitz_minimum_diagnostic':mineig,'floating_point_diagnostics':errors}
        records.append(rec);arrays[f'L{length}_Cn6']=c;arrays[f'L{length}_returns']=a
        previous=c
        print(json.dumps({'L':length,'dimension':length*length,
          'bound':min(2.,float(bound)),'old_difference':difference}),flush=True)
    npz=OUT/'COMMON_CYCLIC_CAR_KATO_MOMENTS.npz';np.savez_compressed(npz,**arrays)
    result={'status':'COMMON_REGULATOR_COUPLED_U1_KATO_AND_GENUINE_CAR_CONSTRUCTION',
      'actual_plaquettes':certificate['actual_plaquette_ids'],
      'Gauss':gauss_check(data),'CAR_modes':6,'CAR_dimension':64,
      'total_charge_by_occupation':charges,'CAR_identity_checks_exact':True,
      'source_Gram_exact':'I6',
      'common_Hilbert_space':'l2(Z_L^2) tensor F_CAR(C^6), in affine Gauss charts E=B(r,s)+Q*h with dual endpoint.',
      'primitive_selection':'Unique ground of the block-diagonal unwrapped representative cost KE_Q+s KB; the total-CAR ground is neutral and empty.',
      'state_certificate':'The archived rational radius-ten witness and its full residual apply to every L>=63 because all omitted neighbors lie before the cyclic seam.',
      'neutral_electric_first_excitation_lower':8,
      'nonempty_CAR_electric_minimum_lower':11,
      'uniform_homotopy_gap_lower_rational':str(gap),
      'homotopy_domain':'s in [0,1]; KE and KB are nonnegative; the same witness Rayleigh bound is maximal at s=1.',
      'regulator_convention':'Centered integer cycle representatives in the primitive cost; exact cyclic Floquet gates at L=63*M. No U1 to Z_L gauge-group replacement.',
      'convergence':'At fixed two-plaquette support, compact-resolvent primitive projectors converge as L grows; explicit moment alias bounds accompany the four regulators.',
      'source_scope':'The U1 components of the six archived leptonic labels, with static source and compensator. Weak/color, receiver streams and physical chiral identification are absent.',
      'cutoffs':records,'old_primitive_certificate_sha256':sha(old_cert),
      'old_data_sha256':sha(old_data),'floating_point_interval_certified':False,
      'full_C10_contract_completed':False,'connected_spatial_limit_certified':False,
      'data_sha256':sha(npz)}
    (OUT/'COMMON_CYCLIC_CAR_KATO_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'uniform_gap':float(gap),'final_bound':records[-1]['infinite_flux_state_and_alias_moment_bound_upper'],
      'full_C10':False},indent=2))


if __name__=='__main__':main()
