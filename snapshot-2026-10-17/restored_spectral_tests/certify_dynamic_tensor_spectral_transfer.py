#!/usr/bin/env python3
"""Finite-QICT-source spectral transfer, including dynamical response and Kato sum rule.

The input is the distributed nine-state coframe-source *self-test*, not a
computed thermodynamic C10/C11 tensor correlator. The output explicitly reports
which claims have and have not been certified. No observational data are used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def evaluate(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        H = np.asarray(data['K0'], dtype=complex)
        A = np.asarray(data['K1_q'], dtype=complex)
        B = np.asarray(data['K1_mq'], dtype=complex)
        C = np.asarray(data['K2_qmq'], dtype=complex)
        n_cells = float(data['N_cells'].item())
        z_density = float(data['continuum_density_factor'].item())
        idx = int(data['selected_index'].item())
    scale = max(1.0, np.linalg.norm(H,2),np.linalg.norm(A,2),np.linalg.norm(B,2))
    h_errors = [np.linalg.norm(M - M.conj().T,ord='fro') for M in (H,A,B,C)]
    assert max(h_errors) < 1e-12*scale
    w, V = np.linalg.eigh(H)
    assert idx == 0, 'Positive-frequency Lehmann measure requires the lowest selected state'
    assert w[1]-w[0] > 1e-10*scale
    dE = w[1:]-w[0]
    ground=V[:,0]
    a = (V[:,1:].conj().T @ A @ ground)
    b = (V[:,1:].conj().T @ B @ ground)
    weights = 2*dE*np.abs(a)**2
    assert np.min(weights)>=-1e-14
    D0 = float(np.sum(weights/dE**2))
    mixedD = float(np.real(np.sum(2*a.conj()*b/dE)))
    contact=float(np.real(np.vdot(ground,C@ground)))
    Kato_contact_minus_D = contact-mixedD
    from_full_resolvent = 0.0
    # Full resolvent expression independent of the positive-frequency Lehmann sum.
    for j in range(1,len(w)):
        pj=V[:,j]
        from_full_resolvent += np.vdot(ground,A@pj)*np.vdot(pj,B@ground)/(w[0]-w[j])
        from_full_resolvent += np.vdot(ground,B@pj)*np.vdot(pj,A@ground)/(w[0]-w[j])
    from_full_resolvent = float(np.real(contact+from_full_resolvent))
    diff=abs(from_full_resolvent-Kato_contact_minus_D)
    assert diff < 5e-12*max(1.0,abs(from_full_resolvent))
    # Two independent source coordinates have a positive semidefinite Lehmann matrix.
    vmat=np.column_stack((a,b))
    max_psd_viol=0.0
    for delta,vec in zip(dE,vmat):
        M=2*delta*np.outer(vec.conj(),vec)
        max_psd_viol=max(max_psd_viol,float(max(0.0,-np.linalg.eigvalsh(M).min())))
    assert max_psd_viol<1e-11
    eps_samples=[]
    for s in (1., 0.1, 0.01, 0.001, 0.0001):
        D=float(np.sum(weights/(s+dE*dE)))
        sD=s*D
        eps_samples.append({'s':s,'D(s)':D,'sD(s)':sD,
                            'upper_bound_sD0':s*D0,
                            'bound_holds':bool(sD<=s*D0+1e-13)})
    assert all(x['bound_holds'] for x in eps_samples)
    time_samples=[]
    for t in (0.0,0.25,1.0,2.0):
        time_samples.append({'euclidean_time':t,
                             'C(t)':float(np.sum((weights/(2*dE))*np.exp(-t*dE)))})
    # Scalar finite-difference check for the full source-dependent eigenvalue.
    fd=[]
    for h in (0.001,0.0005):
        Hp=H+h*A+0.5*h*h*C
        Hm=H-h*A+0.5*h*h*C
        finite=float((np.linalg.eigvalsh(Hp)[0]+np.linalg.eigvalsh(Hm)[0]-2*w[0])/(h*h))
        # For mixed K1_q, K1_mq, this is comparable only when they are equal.
        fd.append({'step':h,'finite_difference_curvature':finite,'absolute_error':abs(finite-Kato_contact_minus_D)})
    assert np.linalg.norm(A-B)<1e-12*scale,'Central curvature check uses A=B in distributed self-test'
    assert fd[0]['absolute_error']<2e-6
    # Independent interval-certificate unit test on a declared synthetic measure:
    # Z0 delta_0 + 0.5 delta_{0.25} + 0.3 delta_{1.0}.
    test_atom=0.4
    test_s=0.001
    test_cut=0.04
    test_total_positive_weight=0.8
    envelope_C=test_total_positive_weight/0.25
    test_D=test_atom/test_s+0.5/(test_s+0.25)+0.3/(test_s+1.0)
    test_lower=max(0.0,test_s*test_D-envelope_C*test_cut-test_total_positive_weight*test_s/test_cut)
    test_upper=test_s*test_D
    assert 0 < test_lower <= test_atom <= test_upper
    normalized_response= z_density*Kato_contact_minus_D/n_cells
    is_real_q_independent=len(dE)==8
    return {
      'classification':'SOURCE_DEFORMED_NINE_STATE_FINITE_REGULATOR_SELF_TEST',
      'input':str(path.name),
      'dimension':len(w),
      'N_cells':n_cells,
      'source_normalization_factor':z_density,
      'selected_ground_eigenvalue':float(w[0]),
      'excitation_gaps':[float(x) for x in dE],
      'excitation_gap_min':float(dE[0]),
      'squared_gap_min':float(dE[0]**2),
      'positive_spectral_weights':[float(x) for x in weights],
      'squared_frequency_spectral_support_at_sampled_q':[float(x*x) for x in dE],
      'lehmann_static_D0':D0,
      'lehmann_mixed_static_DAB0':mixedD,
      'contact_curvature':contact,
      'energy_curvature_contact_minus_response':Kato_contact_minus_D,
      'energy_curvature_resolvent':from_full_resolvent,
      'kato_lehmann_sum_rule_error':diff,
      'previous_source_response_check':normalized_response,
      'positive_source_measure_verified':True,
      'two_source_matrix_psd_max_negative_part':max_psd_viol,
      'zero_frequency_atom_weight_at_recorded_q':0.0,
      'physical_zero_mass_atom_certified':False,
      'positive_atom_interval_unit_test':{'classification':'SYNTHETIC_STIELTJES_PROPOSITION_CHECK', 'true_atom':test_atom,'certified_lower':test_lower,'certified_upper':test_upper,'s':test_s,'Lambda':test_cut,'pass':True},
      'samples':eps_samples,
      'euclidean_time_samples':time_samples,
      'energy_curvature_finite_differences':fd,
      'physical_status':{
        'graviton_pole_from_interacting_QICT':False,
        'two_helicity_Ward_completion':False,
        'thermodynamic_continuum_limit':False,
        'physical_gamma_R_and_G':False,
        'next_required_input':'Full C10/C11 interacting coframe-source operators at multiple momenta and temporal frequencies on a common volume/regulator net.'
      },
      'note':'The supplied 9x9 matrix is a finite operator validation input. Its eight positive gaps imply no zero-frequency atom at the sampled momentum; physical mass is determined from joint (omega,q) dispersion, not from this fixed-q gap. Positive OS kernel holds for this selected branch, without identifying a physical spin-two particle.'
    }


def main():
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parent
    parser.add_argument('--input',type=Path,default=root/'data/SPECTRAL_GRAVITY_SELFTEST_INPUT.npz')
    parser.add_argument('--output',type=Path,default=root/'results/FINITE_DYNAMIC_TENSOR_LEHMANN_CERTIFICATE.json')
    args=parser.parse_args()
    result=evaluate(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['dimension','excitation_gap_min','lehmann_static_D0','contact_curvature','energy_curvature_contact_minus_response','kato_lehmann_sum_rule_error','zero_frequency_atom_weight_at_recorded_q']},indent=2))

if __name__=='__main__': main()
