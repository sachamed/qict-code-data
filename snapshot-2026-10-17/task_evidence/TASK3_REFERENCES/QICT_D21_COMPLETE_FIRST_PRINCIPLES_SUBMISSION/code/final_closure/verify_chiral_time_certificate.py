#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'blind_execution'/'protected_zero_chiral'/'UNITARY_CMV_CHIRAL_TIME_CERTIFICATE.json'
d=json.loads(p.read_text())
cmp=d['comparisons']['19_to_21']
maxB=max(x['relative_change'] for x in cmp['B'])
rank6=d['rank6_free_chiral_certificate']
ph=d['photon_protected_zero']
last=d['cases'][-1]
ok=(maxB<1e-12 and rank6['unitarity_residual']<1e-12 and rank6['LR_block_recovery_residual']<1e-12 and abs(ph['omega_at_zero'])<1e-15 and max(last['D_ratio_to_smallest'])<2.0)
out={'status':'PASS' if ok else 'FAIL','max_B_relative_change_19_to_21':maxB,'rank6_unitarity_residual':rank6['unitarity_residual'],'rank6_LR_recovery_residual':rank6['LR_block_recovery_residual'],'photon_zero':ph['omega_at_zero'],'family_only_D_ratios':last['D_ratio_to_smallest'],'interpretation':'Numerical convergence passes; family-only attenuation is explicitly rejected as the lepton mass kernel.'}
print(json.dumps(out,indent=2))
raise SystemExit(0 if ok else 1)
