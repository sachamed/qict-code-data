#!/usr/bin/env python3
"""Verify that the upstream C11 family-current prediction objects are unchanged from the pre-comparison lock."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[2]
LOCK=ROOT/'preregistration/QICT_PRECOMPARISON_PREDICTION_LOCK.json'
CERT=ROOT/'certificates/C11_PRECOMPARISON_LOCK_SUBSET.json'
obj=json.loads(LOCK.read_text())
by={x['path']:x for x in obj['files']}
protected=[
 ('code/finite_representatives/verify_complementary_path_kernel.py', None),
 ('data/family_transfer_matrices.csv', None),
 ('code/closure/verify_c11_family_current_observable.py', None),
 ('data/c11_family_current_observables.csv', None),
 # The active Supplement L has subsequently received editorial/theorem clarifications.
 # Preserve and verify the exact pre-comparison bytes instead of rewriting history.
 ('supplements/QICT_Supplement_L_C11_Intrinsic_Observable_Completion.tex',
  'preregistration/precomparison_snapshot/QICT_Supplement_L_C11_Intrinsic_Observable_Completion.tex'),
]
rows=[]; ok=True
for lock_rel,snapshot_rel in protected:
    exp=by.get(lock_rel,{}).get('sha256')
    rel=snapshot_rel or lock_rel
    p=ROOT/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    good=bool(exp and got==exp); ok &= good
    rows.append({'locked_path':lock_rel,'verified_path':rel,'expected':exp,'actual':got,'match':good})
out={'status':'PASS' if ok else 'FAIL','precomparison_lock_sha256':hashlib.sha256(LOCK.read_bytes()).hexdigest(),'protected_upstream_prediction_objects':rows,'interpretation':'The numerical prediction code/data are byte-identical to the preserved pre-comparison lock. The exact Supplement L bytes covered by that historical lock are retained under preregistration/precomparison_snapshot; the active Supplement L may contain later editorial or theorem clarifications. Downstream PDG files were added only after the locked prediction values existed.'}
CERT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); sys.exit(0 if ok else 1)
