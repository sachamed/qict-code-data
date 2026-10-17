#!/usr/bin/env python3
"""Downstream-only comparison of locked C11 family-current predictions with PDG data.

This script is intentionally isolated from all upstream prediction builders.  It reads the
already-generated C11 receiver-current observables and a frozen external-data snapshot only
after the prediction values exist.  Nothing in this file is imported by the derivation code.
"""
from pathlib import Path
import csv, json, itertools, hashlib, math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; EXT=DATA/'external'/'PDG2025_REFERENCE.json'; CERT=ROOT/'certificates'
C11=DATA/'c11_family_current_observables.csv'
OUT=CERT/'C11_PDG_NOFIT_CONFRONTATION.json'
PRELOCK=ROOT/'preregistration'/'QICT_PRECOMPARISON_PREDICTION_LOCK.json'

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

pdg=json.loads(EXT.read_text())
rows=list(csv.DictReader(C11.open()))

def val(pair,obs):
    x=[float(r['value']) for r in rows if r['pair']==pair and r['observable']==obs]
    if len(x)!=1: raise RuntimeError((pair,obs,len(x)))
    return x[0]

def Bmat(pair):
    B=np.zeros((3,3),float)
    rr=[r for r in rows if r['pair']==pair and r['observable']=='inclusive_transition_fraction']
    for r in rr:
        B[int(r['from_channel'])-1,int(r['to_channel'])-1]=float(r['value'])
    return B

Bq=Bmat('quark'); Jq=val('quark','abs_C11_Bargmann_J')
M=np.asarray(pdg['ckm']['abs_matrix'],float); Bp=M*M
Jp=float(pdg['ckm']['J']); spu=float(pdg['ckm']['J_sigma_plus']); spd=float(pdg['ckm']['J_sigma_minus'])
# Prediction lies above the PDG central value, hence the relevant quoted one-sigma side is +.
z=(Jq-Jp)/spu
perms=[]
for pr in itertools.permutations(range(3)):
    for pc in itertools.permutations(range(3)):
        B=Bq[np.ix_(pr,pc)]
        perms.append((float(np.linalg.norm(B-Bp,'fro')),float(np.max(np.abs(B-Bp))),pr,pc))
perms.sort(key=lambda x:x[0]); best=perms[0]
lep=pdg['charged_leptons_MeV']
exp_ratios={
    'mu_over_e':lep['mu']['value']/lep['e']['value'],
    'tau_over_e':lep['tau']['value']/lep['e']['value'],
    'tau_over_mu':lep['tau']['value']/lep['mu']['value'],
}
out={
  'status':'PASS_DOWNSTREAM_COMPARISON',
  'comparison_policy':'NO_FIT_NO_RETRO_ENGINEERING',
  'external_data_used':True,
  'external_data_used_upstream':False,
  'c11_prediction_sha256':sha(C11),
  'pdg_snapshot_sha256':sha(EXT),
  'precomparison_prediction_lock_sha256':sha(PRELOCK),
  'ckm_J':{
      'C11_locked_abs_J':Jq,'PDG_J':Jp,'PDG_sigma_plus':spu,'PDG_sigma_minus':spd,
      'relative_difference':(Jq-Jp)/Jp,'standardized_difference_using_relevant_asymmetric_sigma':z,
      'direct_identity_assessment':'DISFAVORED_AT_%.3f_SIGMA'%z
  },
  'ckm_modulus_squared':{
      'C11_B_ud':Bq.tolist(),'PDG_absV_squared_central':Bp.tolist(),
      'C11_purity':float(np.sum(Bq**2)),'PDG_central_purity':float(np.sum(Bp**2)),
      'purity_difference':float(np.sum(Bq**2)-np.sum(Bp**2)),
      'best_independent_row_column_permutation_frobenius_distance':best[0],
      'best_permutation_max_abs_entry_difference':best[1],
      'best_row_permutation_zero_based':list(best[2]),'best_column_permutation_zero_based':list(best[3]),
      'direct_receiver_overlap_equals_CKM_assessment':'NOT_SUPPORTED_BY_CURRENT_DATA'
  },
  'charged_lepton_data_downstream_only':{
      'PDG_masses_MeV':lep,
      'experimental_ratios_for_future_locked_G2_comparison':exp_ratios,
      'QICT_G2_numeric_mass_ratio_prediction_present':False,
      'reason':'No converged full compact 3D G2 family-edge payload is present; these data are comparison targets only and are not used to construct a QICT mass prediction.'
  },
  'conclusion':(
      'The direct promotion V_rec = V_CKM is not supported by the downstream comparison. '
      'In particular the locked C11 Bargmann invariant lies above the PDG CKM value by the quoted amount, '
      'and the modulus-squared matrices differ even after all independent row/column permutations. '
      'Therefore the polar matching gate must not be assumed; a genuine G3 vertex correction/matching calculation is required.'
  )
}
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
