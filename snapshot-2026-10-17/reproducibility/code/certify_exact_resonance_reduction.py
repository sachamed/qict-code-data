#!/usr/bin/env python3
"""Exact 63-cycle Bloch reduction and charged U1 GNS moments.

The full ordered C10 operator is not replaced by this rotor component.
Arithmetic proofs are separate from floating-point FFT diagnostics.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from nonabelian_common_car_kato import polynomial_tail,sqrt_upper

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/exact_resonance'
PAYLOAD=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
PRIOR=ROOT/'reproducibility/results/nonabelian_common_car'
PHI=math.pi/63;PERIOD=63;DEPTH=12


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def polynomial_divide(a,b):
    a=list(a);b=list(b)
    while len(a)>1 and a[-1]==0:a.pop()
    quotient=[0]*max(1,len(a)-len(b)+1)
    while len(a)>=len(b) and any(a):
        assert a[-1]%b[-1]==0
        k=len(a)-len(b);v=a[-1]//b[-1];quotient[k]=v
        for j,c in enumerate(b):a[k+j]-=v*c
        while len(a)>1 and a[-1]==0:a.pop()
    return quotient,a


def cyclotomic(n,cache):
    if n in cache:return cache[n]
    p=[-1]+[0]*(n-1)+[1]
    for d in range(1,n):
        if n%d==0:
            p,remainder=polynomial_divide(p,cyclotomic(d,cache))
            assert not any(remainder)
    cache[n]=p;return p


def gauss_certificate(charge):
    counts=[0]*PERIOD
    for r in range(PERIOD):counts[(-4*r*r-2*charge*r)%PERIOD]+=1
    norm=[sum(counts[i]*counts[(i-k)%PERIOD] for i in range(PERIOD))
          for k in range(PERIOD)]
    _,remainder=polynomial_divide(norm,cyclotomic(PERIOD,{}))
    assert remainder==[PERIOD]
    return {'qY':charge,'integer_residue_counts':counts,
            'squared_Gauss_modulus_cyclotomic_remainder':remainder}


def normalized_witness(record):
    w=[F(x) for x in record['rational_witness_components']]
    norm=sum(x*x for x in w)
    return np.array([float(x) for x in w])/math.sqrt(float(norm))


def fiber_returns(labels,seed,samples,charge):
    """Angle-domain 63-dimensional fibers; pure returns averaged first."""
    theta=2*np.pi*np.arange(samples)/samples
    coeff=np.zeros((samples,PERIOD),complex)
    for r,value in zip(labels,seed):
        coeff[:,int(r)%PERIOD]+=value*np.exp(1j*r*theta/PERIOD)
    initial=np.fft.ifft(coeff,axis=1,norm='ortho');state=initial.copy()
    j=np.arange(PERIOD)
    electric=8*j*j+4*charge*j+2*charge*charge+charge
    phase_e=np.exp(-1j*PHI*(electric%126))
    angle=theta[:,None]/PERIOD+2*np.pi*j[None,:]/PERIOD
    phase_b=np.exp(-1j*PHI*(4-4*np.cos(angle)))
    returns=[];norms=[]
    for n in range(DEPTH+1):
        returns.append(np.einsum('mj,mj->m',initial.conj(),state))
        norms.append(float(np.max(abs(np.sum(abs(state)**2,axis=1)-1))))
        state=phase_b*np.fft.ifft(phase_e*np.fft.fft(state,axis=1,norm='ortho'),
                                axis=1,norm='ortho')
    return np.asarray(returns).T,np.mean(returns,axis=1),max(norms)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(PAYLOAD,allow_pickle=False) as z:
        edges=z['plaquette_edge_indices'].astype(int)
        signs=z['plaquette_edge_sign'].astype(int)
        test_flux=np.arange(len(z['links']),dtype=np.int64)%11-5
    cycle_sums=signs.sum(axis=1);assert np.all(cycle_sums==0)
    dot=np.sum(test_flux[edges]*signs,axis=1)
    norm=np.sum(signs*signs,axis=1)
    delta=4*63*dot+2*63**2*norm+63*cycle_sums
    assert np.all(delta%126==0)
    rejected=[m for m in range(1,63) if (16*m)%126!=0]
    assert len(rejected)==62
    primitive=json.loads((PRIOR/'NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json').read_text())
    ground=primitive['cutoff_results'][-1]['ground_certificates']['U1']
    labels=np.array(ground['support_labels'],int).ravel()
    seed=normalized_witness(ground)
    state_bound=sqrt_upper(F(2))*F(ground['state_trace_distance_upper_rational'])
    arrays={};records=[];previous=None
    with np.load(PRIOR/'NONABELIAN_COMMON_CAR_KATO_MOMENTS.npz',allow_pickle=False) as z:
        old_returns=np.array([z['L3_neutral_group_returns'][0],
                              *z['L3_charged_group_returns'][:2]])
    for samples in [1,2,4,8,16]:
        all_ret=[];point=[];norm_errors=[]
        for charge in [0,-3,-6]:
            p,a,e=fiber_returns(labels,seed,samples,charge)
            point.append(p);all_ret.append(a);norm_errors.append(e)
        a=np.asarray(all_ret)
        c=np.array([np.diag([a[0,n].conj()*a[1,n]]*3+
                            [a[0,n].conj()*a[2,n]]*3) for n in range(DEPTH+1)])
        length=63*samples;radius=int(np.max(abs(labels)))
        order=(length-2*radius-1)//DEPTH
        tail=polynomial_tail(F(1,5),order)
        bound=4*state_bound+4*DEPTH*tail*(1+tail)**(DEPTH-1)
        toeplitz=np.block([[c[j-i] if j>=i else c[i-j].conj().T
                           for j in range(DEPTH+1)] for i in range(DEPTH+1)])
        mineig=float(np.linalg.eigvalsh(toeplitz).min())
        assert mineig>-1e-12 and np.linalg.norm(c[0]-np.eye(6))<1e-12
        error=float(np.max(abs(a-old_returns)))
        assert error<2e-12
        records.append({'Bloch_samples':samples,'fiber_dimension':63,
          'equivalent_cyclic_flux_dimension':length,'polynomial_order':order,
          'analytic_state_and_quadrature_bound_rational':str(bound),
          'analytic_state_and_quadrature_bound_upper':min(2.,float(bound)),
          'uncapped_analytic_bound_upper':float(bound),
          'successive_moment_change':None if previous is None else float(np.max(abs(c-previous))),
          'independent_buffered_return_difference':error,
          'C0_Frobenius_error':float(np.linalg.norm(c[0]-np.eye(6))),
          'Toeplitz_minimum_diagnostic':mineig,
          'maximum_fiber_norm_error':max(norm_errors)})
        arrays[f'M{samples}_Cn6']=c;arrays[f'M{samples}_group_returns']=a
        arrays[f'M{samples}_point_returns']=np.asarray(point)
        previous=c
        print(json.dumps({'M':samples,'bound':min(2.,float(bound)),
                          'buffered_difference':error}),flush=True)
    pi_cert=json.loads((ROOT/'reproducibility/results/spatial_state/SPATIAL_STATE_SELECTION_CERTIFICATE.json').read_text())
    pi_lo,pi_hi=map(F,pi_cert['exact_U1_primitive_return_certificate']['pi_interval_rational'])
    xlo,xhi=4*pi_lo/63,4*pi_hi/63
    leading=(xlo/2)**63/math.factorial(63)
    lower=leading*(1-xhi*xhi/(4*64))
    upper=(xhi/2)**63/math.factorial(63)
    assert lower>0
    beta_records=[]
    for charge in [0,-3,-6]:
        beta=F(charge,4);nu=8*(beta-F(63,2))
        assert nu.denominator==1 and math.gcd(8,63)==1
        beta_records.append({'qY':charge,'beta_rational':str(beta),
          'nu_integer':int(nu),'tau_over_2pi':'8/63',
          'kick_strength_mu':'-4*pi/63','primitive_resonance_length':63})
    npz=OUT/'EXACT_BLOCH_U1_COMMON_GNS_MOMENTS.npz'
    np.savez_compressed(npz,**arrays)
    result={'status':'EXACT_63_CYCLE_REDUCTION_AND_EXECUTED_INFINITE_ROTOR_MOMENTS',
      'payload_sha256':sha(PAYLOAD),'plaquettes_checked':len(edges),
      'global_neutral_cycle_rank':13356,
      'exact_translation_period':63,
      'period_identity':'(K_E(E+63*d)-K_E(E))/126=2*E.dot(d)+63*||d||^2 for sum(d)=0.',
      'all_smaller_single_loop_periods_rejected':rejected,
      'minimal_period_scope':'Any period on the full neutral carrier must restrict to the four-link primitive loop; its r-dependent phase forces 63 to divide the period.',
      'full_C10_scope':'The Abelian cycle reduction applies under the previously stated configuration-wise factor types. It does not supply the uncomputed charged stream, route-first interaction, chiral embedding or selected exterior state.',
      'gauss_sum_certificates':[gauss_certificate(q) for q in [0,-3,-6]],
      'resonant_rotor_parameter_mapping':beta_records,
      'spectral_result':{'basis':'Guarneri (2009), Theorem 1, after the exact mapping above and magnetic/electric order conjugacy.',
        'doi':'10.1007/s00023-009-0008-9','arxiv':'0902.3896',
        'rotor_spectrum':'purely absolutely continuous',
        'normal_stationary_trace_one_density_on_this_exact_rotor_exists':False,
        'scope':'Isolated exact U1 plaquette, including qY=0,-3,-6 static-source modules; no such spectral conclusion is transferred to coupled/full C10.'},
      'first_Bloch_trace_harmonic':{'Bessel_order':63,
        'Bessel_J63_lower_rational':str(lower),'Bessel_J63_upper_rational':str(upper),
        'Bessel_J63_lower':float(lower),'Bessel_J63_upper':float(upper),
        'trace_coefficient_modulus_lower_rational':str(7*lower),
        'trace_coefficient_modulus_upper_rational':str(8*upper),
        'scope':'A strict trace harmonic, not a bound on each individual bandwidth; prime-order bandwidth formulas are not applied at order 63.'},
      'pure_GNS_Bloch_type':'Two independent Bloch arguments on HS(H0,Hf); average each return before multiplying. A single diagonal fiber is not silently substituted.',
      'wrong_single_fiber_normalization_witness':{'neutral_state':'(|0>+|63>)/sqrt(2)',
        'correct_C0':'1','unrenormalized_diagonal_only_HS_C0':'3/2',
        'scope':'Admissible neutral state proving source-carrier distinction, not a second A7 ground.'},
      'source_Gram_exact':'I6 on the stated CAR vacuum and affine Wilson dressing',
      'cutoffs':records,'floating_point_interval_certified':False,
      'full_C10_contract_completed':False,'connected_spatial_limit_certified':False,
      'data_sha256':sha(npz)}
    (OUT/'EXACT_RESONANCE_REDUCTION_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'Bessel_J63_interval':[float(lower),float(upper)],
      'final_bound':records[-1]['analytic_state_and_quadrature_bound_upper'],
      'full_C10':False},indent=2))


if __name__=='__main__':main()
