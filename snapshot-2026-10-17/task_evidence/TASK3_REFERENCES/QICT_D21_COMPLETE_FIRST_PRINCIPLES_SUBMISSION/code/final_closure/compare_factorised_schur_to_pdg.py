#!/usr/bin/env python3
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parents[2]
PRED=json.loads((ROOT/'predictions'/'FACTORISED_SCHUR_PRECOMPARISON_LOCK.json').read_text())
OUT=ROOT/'results'/'factorised_schur'
# PDG 2026 values are read only in this post-comparison program.
me=0.51099895069
mmu=105.6583755
mtau=1776.93
obs={'m_mu_over_m_e':mmu/me,'m_tau_over_m_mu':mtau/mmu,'m_tau_over_m_e':mtau/me}
pr=PRED['predictions']
comparison={
 'status':'POSTCOMPARISON_ONLY',
 'prediction_lock_sha256':(ROOT/'predictions'/'FACTORISED_SCHUR_PRECOMPARISON_LOCK.sha256').read_text().split()[0],
 'pdg_2026_masses_MeV':{'electron':me,'muon':mmu,'tau':mtau},
 'observed_ratios':obs,
 'predicted_ratios':{k:pr[k] for k in obs},
 'relative_errors_percent':{k:100*(pr[k]/obs[k]-1) for k in obs},
 'one_scale_electron_calibration_MeV':{
    'electron_input':me,
    'muon_prediction':me*pr['m_mu_over_m_e'],
    'tau_prediction':me*pr['m_tau_over_m_e']
 },
 'koide_Q_prediction':pr['koide_Q_from_predicted_ratios'],
 'koide_Q_observed':(me+mmu+mtau)/(math.sqrt(me)+math.sqrt(mmu)+math.sqrt(mtau))**2,
 'alpha_inferred_separately':{
    'from_mu_over_e':math.log(obs['m_mu_over_m_e'])/34.0,
    'from_tau_over_mu':math.log(obs['m_tau_over_m_mu'])/18.0
 },
 'claim_boundary':'No experimental mass enters compute_factorised_chiral_schur_prediction.py. This file is a post-comparison audit only.'
}
comparison['alpha_inferred_mean']=0.5*(comparison['alpha_inferred_separately']['from_mu_over_e']+comparison['alpha_inferred_separately']['from_tau_over_mu'])
comparison['alpha_prediction']=PRED['factorization_rule']['alpha_numeric']
comparison['alpha_relative_gap_percent']=100*(comparison['alpha_prediction']/comparison['alpha_inferred_mean']-1)
(OUT/'POST_PDG_FACTORISED_SCHUR_COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n')
print(json.dumps(comparison,indent=2))
