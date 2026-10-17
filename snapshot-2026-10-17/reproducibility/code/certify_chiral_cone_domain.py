#!/usr/bin/env python3
"""Exact free-BCC overlap tails and an allowed covariant-kernel gap witness.

The background witness uses the Hermitian nearest-neighbor covariantization
of the displayed C7 Fourier stencil. It establishes that the displayed
unitary-link assumptions do not imply a uniform Wilson-kernel gap. It does
not assert that a separately specified domain-wall update has been evaluated.
"""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import hashlib
import itertools
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reproducibility/results/chiral_cone_domain'
ACTION = ROOT/'reproducibility/data/physical_g2_inputs/L24_RANK6_STATE_CONTRACTION_PAYLOAD_2026-10-04/components/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json'


def binomial_coefficient(n):
    return F(math.comb(2*n,n),4**n)


def inverse_root_coefficient(m, terms=80):
    m=abs(m)
    value=sum((binomial_coefficient(n)*binomial_coefficient(n+m)
               *F(1,2**(2*n+m)) for n in range(terms)),F(0))
    # Each omitted binomial factor is at most one.
    tail=F(1,2**m)*F(1,4**terms)*F(4,3)
    return value,value+tail


def covariant_symbol(k, charge=-3, t=math.pi/3):
    """Replace each free Fourier shift by its Hermitian BCC link transport."""
    k=np.asarray(k,float)
    w=np.ones(k.shape[:-1],complex)
    n=np.zeros(k.shape[:-1]+(3,),complex)
    for sx,sy,sz in itertools.product((-1,1),repeat=3):
        sig=np.array([sx,sy,sz])
        # The four forward phases are t*sy*sz in representation qY;
        # inverse links carry the adjoint. Gauge-link angles are divided by qY.
        alpha=(sx*sy*sz)*t/charge
        shift=np.exp(1j*(k@sig+charge*alpha))
        w-=shift/8
        n[...,0]+=(sx/(8j)-sy*sz/8)*shift
        n[...,1]+=(sy/(8j)+sx*sz/8)*shift
        n[...,2]+=(sz/(8j)-sx*sy/8)*shift
    assert np.max(abs(w.imag))<5e-15 and np.max(abs(n.imag))<5e-15
    return n.real,w.real


def clifford():
    s=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),
       np.diag([1,-1]).astype(complex)]
    gamma=[np.kron(s[1],a) for a in s]
    gamma5=np.kron(s[2],np.eye(2))
    for a in gamma:
        assert np.linalg.norm(a@a-np.eye(4))<1e-14
        assert np.linalg.norm(a@gamma5+gamma5@a)<1e-14
    return gamma,gamma5


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    action=json.loads(ACTION.read_text())
    assert action['fermion_discretization']['M0_over_rW']==.5
    gamma,g5=clifford()
    sample=np.array([[.13,.27,.41],[-.38,.72,-.29],[0,0,0]])
    n,w=covariant_symbol(sample,t=0)
    c=np.cos(sample);s=np.sin(sample)
    expected=np.column_stack([s[:,0]*c[:,1]*c[:,2]+c[:,0]*s[:,1]*s[:,2],
        c[:,0]*s[:,1]*c[:,2]-s[:,0]*c[:,1]*s[:,2],
        c[:,0]*c[:,1]*s[:,2]+s[:,0]*s[:,1]*c[:,2]])
    free_error=max(float(np.max(abs(n-expected))),float(np.max(abs(w-(1-np.prod(c,axis=1))))))
    assert free_error<1e-14
    records=[]; coefficients=[]
    for radius in [4,8,10,12,16,24,32,48]:
        m=radius+1
        rprev=inverse_root_coefficient(m-1);rnow=inverse_root_coefficient(m)
        p_tail_lower=(rprev[0]-rnow[1]/2)/2
        p_tail_upper=(rprev[1]-rnow[0]/2)/2
        simple_lower=(binomial_coefficient(radius)-binomial_coefficient(radius+1)/4)*F(1,2**(radius+1))
        assert p_tail_lower>=simple_lower>0
        # This is the restriction to k=(k_x,0,0), not a bound for a 3D tail.
        uniform_line_tail=F(8,3)*F(1,2**radius)
        records.append({'coordinate_range':radius,
          'first_omitted_line_projector_coefficient_interval':[str(p_tail_lower),str(p_tail_upper)],
          'unavoidable_uniform_operator_error_lower_rational':str(simple_lower),
          'unavoidable_uniform_operator_error_lower':float(simple_lower),
          'line_fourier_truncation_upper_rational':str(uniform_line_tail),
          'line_fourier_truncation_upper':float(uniform_line_tail)})
    for m in range(-64,65):
        rm=inverse_root_coefficient(m);rp=inverse_root_coefficient(m+1)
        coefficients.append(float((rm[0]+rm[1])/4-(rp[0]+rp[1])/2))
    nk=16384;k=2*np.pi*np.arange(nk)/nk
    f=(.5-np.exp(-1j*k))/np.sqrt(1.25-np.cos(k))
    fft=np.fft.fft(f)/nk
    ferr=max(abs(fft[m%nk]-coefficients[m+64]) for m in range(-32,33))
    assert ferr<2e-14
    # Exact cancellations at the rough-field witness: cos(t)=1/2 and
    # sin(t) coefficients cancel by the four forward sign characters.
    forwards=[(1,y,z) for y,z in itertools.product((-1,1),repeat=2)]
    exact_sums={'cos_sum':sum(F(1,2) for _ in forwards),
                'sin_character_sum':sum(y*z for _,y,z in forwards),
                'y_sin_character_sum':sum(z for _,y,z in forwards),
                'z_sin_character_sum':sum(y for _,y,z in forwards),
                'yz_cos_character_sum':sum(F(y*z,2) for _,y,z in forwards),
                'y_cos_character_sum':sum(F(y,2) for _,y,z in forwards),
                'z_cos_character_sum':sum(F(z,2) for _,y,z in forwards)}
    assert exact_sums['cos_sum']==2 and all(v==0 for key,v in exact_sums.items() if key!='cos_sum')
    gap_records=[]
    reciprocal=np.pi*np.array([[1,1,0],[1,0,1],[0,1,1]],float)
    for L in [2,3,4,6,8]:
        momenta=np.array(list(itertools.product(range(L),repeat=3)))@reciprocal/L
        ni,wi=covariant_symbol(momenta)
        gap=np.sqrt(np.sum(ni*ni,axis=1)+(wi-.5)**2)
        gap_records.append({'primitive_periodic_volume':L,'kernel_blocks':L**3,
                           'minimum_kernel_singular_value':float(gap.min())})
        assert gap.min()<3e-15
    ni,wi=covariant_symbol(np.zeros((1,3)))
    dw=sum(1j*a*ni[0,i] for i,a in enumerate(gamma))+(wi[0]-.5)*np.eye(4)
    herm=float(np.linalg.norm(g5@dw-(g5@dw).conj().T))
    assert np.linalg.norm(dw)<1e-14 and herm<1e-14
    outputs={'coordinate_range':np.array([r['coordinate_range'] for r in records]),
             'line_laurent_labels':np.arange(-64,65),
             'line_laurent_vplus_coefficients':np.array(coefficients),
             'rough_forward_BCC_vectors':np.array(forwards),
             'rough_link_angles_q_minus3':np.array([y*z*math.pi/(-9) for _,y,z in forwards])}
    npz=OUT/'CHIRAL_CONE_DOMAIN_DATA.npz';np.savez_compressed(npz,**outputs)
    direction=[]
    for epsilon in [1e-2,1e-4,1e-6]:
        signs=[]
        for x in [epsilon,-epsilon]:
            n,w=covariant_symbol(np.array([[x,0,0]]))
            D=sum(1j*a*n[0,i] for i,a in enumerate(gamma))+(w[0]-.5)*np.eye(4)
            eigen,vectors=np.linalg.eigh(g5@D)
            signs.append((vectors*np.sign(eigen))@vectors.conj().T)
        direction.append({'kx_magnitude':epsilon,'opposite_direction_sign_difference':float(np.linalg.norm(signs[0]-signs[1],2))})
    assert direction[-1]['opposite_direction_sign_difference']>1.999999
    result={'status':'EXACT_FREE_OVERLAP_RANGE_AND_COVARIANT_KERNEL_DOMAIN_TEST',
      'action_sha256':hashlib.sha256(ACTION.read_bytes()).hexdigest(),
      'source_assumptions':'The displayed C7 free stencil, rW=1, M0=1/2, and Hermitian covariant nearest-neighbor replacement T_-sigma=T_sigma^dagger.',
      'free_axis_H_squared':'5/4-cos(kx)',
      'free_axis_nearest_complex_singularity_distance':'log(2)',
      'free_axis_inverse_root_coefficient':'sum_{n>=0} a_n*a_(n+|m|)*2^(-2n-|m|), a_n=binom(2n,n)/4^n',
      'exact_finite_range_overlap_projector_exists':False,
      'radius_records':records,'independent_FFT_coefficient_error':float(ferr),
      'free_stencil_reconstruction_error':free_error,
      'rough_background':{'representation_charge_qY':-3,
        'transport_scope':'Independent Hermitian nearest-neighbor BCC link transports, as admitted by the displayed kernel conditions; not a reconstruction of the serialized split-route Wilson kernel.',
        'four_forward_link_angles':'alpha_(1,y,z)=-y*z*pi/9; inverse links are adjoints',
        'representation_transport_phases':'qY*alpha=y*z*pi/3',
        'exact_W_at_Gamma':'1/2','exact_n_at_Gamma':[0,0,0],
        'exact_H_W_at_Gamma':'zero 4x4 spin block (times identity on weak indices)',
        'exact_cancellation_records':{k:str(v) for k,v in exact_sums.items()},
        'gauge_cycle':'(1,1,1)+(1,-1,-1)-(1,1,-1)-(1,-1,1)=0',
        'gauge_cycle_holonomy_angle':'-4*pi/9',
        'U1_cycle_magnetic_cost':4*(1-math.cos(4*math.pi/9)),
        'periodic_gap_checks':gap_records,
        'kernel_hermiticity_defect':herm,
        'kernel_sign_jump_norm_across_t_pi_over3':2,
        'Gamma_linear_symbol_jacobian':'diag((1+sqrt(3))/2,(1-sqrt(3))/2,(1+sqrt(3))/2)',
        'opposite_momentum_direction_checks':direction,
        'uniform_exponential_locality_over_this_transport_class':False,
        'uniform_kernel_gap_from_displayed_unitary_link_assumptions':False},
      'restriction_unitarity_identity':'P-P U^dagger P U P=P U^dagger (I-P) U P',
      'required_regulator_link':'A specified local domain-wall update and source map, or a certified interacting overlap-tail and chirality-transport bound on the same carrier.',
      'scope':'A free-kernel range theorem and one compatible covariant-kernel counterexample. This does not rule out an independently specified domain-wall regulator or prove nonlocality for every selected interacting state.',
      'global_C10_charged_operator_certified':False,
      'data_sha256':hashlib.sha256(npz.read_bytes()).hexdigest()}
    (OUT/'CHIRAL_CONE_DOMAIN_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'free_stencil_error':free_error,'FFT_error':float(ferr),
                     'rough_kernel_gap':gap_records[-1]['minimum_kernel_singular_value'],
                     'exact_finite_range_overlap':False},indent=2))


if __name__=='__main__':main()
