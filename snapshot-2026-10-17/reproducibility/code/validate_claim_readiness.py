#!/usr/bin/env python3
"""Independent spectral moment and physical-data admissibility audit.

This checks mathematical consistency and accurate status attribution. It does not
construct physical missing tensors or promote finite witnesses to measurements.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'

def eqnorm(x): return float(np.linalg.norm(x,ord=2))

def moment_model(m:int, modulus:int, L:int):
    z=np.exp(2j*np.pi*(np.arange(L)%modulus)/modulus)
    # 6 source channels with uniform coherent embedding
    return np.array([np.mean(z**n)*np.eye(6,dtype=np.complex128)
                     for n in range(m+2)])

def toeplitz(C,m):
    return np.block([[C[j-k] if j>=k else C[k-j].conj().T
                      for k in range(m+1)] for j in range(m+1)])

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--depth',type=int,default=3)
    args=ap.parse_args()
    m=args.depth
    assert m>=1
    M,N=m+1,m+2
    L=M*N
    A=moment_model(m,M,L); B=moment_model(m,N,L)
    shared=max(eqnorm(A[n]-B[n]) for n in range(m+1))
    separation=eqnorm(A[m+1]-B[m+1])
    Ta=toeplitz(A,m); Tb=toeplitz(B,m)
    eigA=float(np.linalg.eigvalsh(Ta).min())
    eigB=float(np.linalg.eigvalsh(Tb).min())
    assert shared<1e-12 and abs(separation-1)<1e-12 and min(eigA,eigB)>-1e-11
    contractpath=RESULTS/'physical_g2_execution'/'PHYSICAL_G2_CONTRACT_AUDIT.json'
    fitpath=RESULTS/'phenomenology'/'PHENOMENOLOGY_CHI2_AUDIT.json'
    g=json.loads(contractpath.read_text()); f=json.loads(fitpath.read_text())
    expected=set(g['continuum_sequence_inputs'])
    present={k: bool(g['physical_contract_checks'].get(k,False)) for k in expected}
    unready=sorted(k for k in expected if not present[k])
    if g['physical_G2_promoted'] and unready:
        raise AssertionError('a physical G2 claim requires all input classes')
    fit=f['conditional_reference_check']
    n_light=float(fit['observables'][0]['conditional_prediction'])
    obs=float(fit['observables'][0]['measurement']); sig=float(fit['observables'][0]['sigma'])
    stat=((n_light-obs)/sig)**2
    assert abs(stat-float(fit['chi2_conditional']))<1e-10
    assert abs(float(fit['chi2_conditional'])-float(fit['chi2_SM_same_data']))<1e-10
    assert len(unready)==len(set(g['missing_physical_objects']))
    outcome={
      'status':'PASS_MATHEMATICAL_AND_STATUS_CHECKS',
      'classification':'validation of finite moment indeterminacy and truthful continuum status',
      'finite_moment_proof':{
          'shared_hilbert_dimension':int(6*L), 'six_channel_receiver':True,
          'moment_depth':m,'unitary_period_A':M,'unitary_period_B':N,
          'shared_moment_opnorm_residual':shared,
          'next_moment_separation_opnorm':separation,
          'minimum_block_toeplitz_eigenvalues':[eigA,eigB],
          'physical_QICT_spectrum_from_this_example':False
      },
      'physical_G2':{
          'precontraction_certified':str(g['precontraction']['certificate_status']),
          'complete_interacting_input_classes':int(len(expected)),
          'input_classes_available':int(sum(present.values())),
          'input_classes_to_supply':unready,
          'physical_promotion':bool(g['physical_G2_promoted']),
          'certificate_sha256':sha(contractpath),
          'status_class':'finite-regulator precontraction and controls' if unready else 'input-complete; evaluate prescribed limits'
      },
      'likelihood':{
          'independent_observables':int(fit['N_observables']),
          'recomputed_chi2':float(stat),
          'qict_minus_sm_chi2':float(fit['delta_chi2_conditional_minus_SM']),
          'finite_flavor_coordinates_in_global_fit':False,
          'certificate_sha256':sha(fitpath),
          'classification':'conditional reference test; no independent QICT continuum prediction',
          'global_QICT_fit_evaluated':False
      },
      'numerical_error_envelope':{
          'eta_0_example':0.0,'eta_n_example':1e-8,
          'toeplitz_lower_bound_for_m':-(2*m)*1e-8,
          'one_step_delta_example':1e-8,
          'power_error_bound_for_m_plus_1':float((1+1e-8)**(m+1)-1)
      }
    }
    path=RESULTS/'CLAIM_READINESS_AND_MOMENT_IDENTIFIABILITY.json'
    path.write_text(json.dumps(outcome,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print('PASS independent finite six-channel moment test; input classes available:',len(expected)-len(unready),'/',len(expected))
    print('PASS likelihood:', stat, 'data point and common comparator')
    print('PASS source file checks; output:',path)

if __name__=='__main__':main()
