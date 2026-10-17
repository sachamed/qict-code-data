#!/usr/bin/env python3
from pathlib import Path
import json,sys
R=Path(__file__).resolve().parents[1]
T=json.load(open(R/'certificates/PEPO3D_REPRESENTATION_TRUNCATION.json'))
K=json.load(open(R/'certificates/K_CUTOFF_REJECTION.json'))
S=json.load(open(R/'spectral/TASK3_SPECTRAL_METHOD_CERTIFICATE.json'))
G=json.load(open(R/'certificates/TASK3_PHYSICAL_PROMOTION_GATE.json'))
checks={
 'representation_truncation_pass':T['status'].startswith('PASS'),
 'moment_bound_below_2e-9':T['rank6_moment_error_bound_n_le_10']<2e-9,
 'auxiliary_K_rejected':K['status'].startswith('PASS_REJECTION'),
 'spectral_methods_pass':S['status'].startswith('PASS'),
 'no_false_physical_promotion': (not G.get('all_pass_eligible',False)) if 'checks' not in G else (not G['checks'].get('physical_poles_and_residues_promoted',False))
}
out={'status':'PASS_AUDITED_TASK3_LAYERING' if all(checks.values()) else 'FAIL_AUDIT','checks':checks,'physical_promotion_status':G['status']}
print(json.dumps(out,indent=2));sys.exit(0 if all(checks.values()) else 1)
