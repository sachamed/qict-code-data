#!/usr/bin/env python3
from pathlib import Path
import json, math, hashlib
ROOT=Path(__file__).resolve().parents[2]
LOCK=ROOT/'predictions'/'CONNECTED_CAR_PRECOMPARISON_LOCK.json'
OUT=ROOT/'results'/'connected_car'/'POST_PDG_CONNECTED_CAR_COMPARISON.json'
p=json.loads(LOCK.read_text())
# External comparison only; none of these values is read by the prediction program.
me=0.51099895069
mmu=105.6583755
mtau=1776.93
r1=mmu/me; r2=mtau/mmu; r3=mtau/me
q=p['predictions']
a=float(p['alpha_connected'])
a_me=math.log(r1)/34.0
a_tm=math.log(r2)/18.0
amean=(a_me+a_tm)/2.0
out={
 'status':'POSTCOMPARISON_ONLY',
 'prediction_lock_sha256':hashlib.sha256(LOCK.read_bytes()).hexdigest(),
 'PDG_2026_MeV':{'electron':me,'muon':mmu,'tau':mtau},
 'experimental_ratios':{'mu_over_e':r1,'tau_over_mu':r2,'tau_over_e':r3},
 'predicted_ratios':q,
 'relative_errors':{
   'mu_over_e':q['m_mu_over_m_e']/r1-1.0,
   'tau_over_mu':q['m_tau_over_m_mu']/r2-1.0,
   'tau_over_e':q['m_tau_over_m_e']/r3-1.0
 },
 'experimental_log_coefficients':{
   'alpha_from_mu_over_e':a_me,
   'alpha_from_tau_over_mu':a_tm,
   'mean':amean
 },
 'alpha_connected_relative_to_experimental_mean':a/amean-1.0
}
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
