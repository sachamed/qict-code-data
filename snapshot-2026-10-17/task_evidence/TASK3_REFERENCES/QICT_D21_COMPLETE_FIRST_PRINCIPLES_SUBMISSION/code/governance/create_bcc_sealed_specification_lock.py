#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'preregistration'/'QICT_BCC_SEALED_PREDICTION_LOCK.json'
SHA=ROOT/'preregistration'/'QICT_BCC_SEALED_PREDICTION_LOCK.sha256'
FILES=[
 'preregistration/QICT_BCC_SEALED_PREDICTION.md',
 'code/photon_metrology/verify_bcc_cubic_harmonic_fingerprint.py',
 'supplements/QICT_Supplement_X_Assumptions_Global_Gauge_Consistency_and_Convergence.tex',
]
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()
entries=[{'path':rel,'sha256':sha256(ROOT/rel),'bytes':(ROOT/rel).stat().st_size} for rel in FILES]
canonical=''.join(f"{x['path']}\0{x['sha256']}\n" for x in entries).encode()
payload=hashlib.sha256(canonical).hexdigest()
obj={
 'theory':'Quantum Information Copy Time (QICT)',
 'prediction':'BCC cubic-harmonic sky fingerprint',
 'status':'FIXED_LOCAL_SPECIFICATION_DIGEST',
 'scientific_status':'PROVEN_WITHIN_DECLARED_BCC_PHOTON_TANGENT_COMPLETION',
 'external_data_used':False,
 'payload_sha256':payload,
 'files':entries,
 'predictions':{
  'parity':'even','quadratic_order_harmonic_support':['l=0','l=4'],
  'critical_point_topology':'6:12:8','axis_face_body_ratio':'9:21:25',
  'successive_gap_ratio':'3','l4_to_monopole_power_ratio':'1024/20181',
  'centered_skewness':'-18*sqrt(21)/143','centered_kurtosis':'5607/2431',
  'shape_invariant_mu3sq_over_mu2cubed':'6804/20449'},
 'external_timestamp_required_for_independent_preregistration_claim':True,
 'comparison_status':'NO_HELD_OUT_EXPERIMENTAL_COMPARISON_INCLUDED'
}
OUT.write_text(json.dumps(obj,indent=2)+'\n')
SHA.write_text(payload+'  QICT_BCC_SEALED_PREDICTION_LOCK\n')
print(json.dumps(obj,indent=2))
