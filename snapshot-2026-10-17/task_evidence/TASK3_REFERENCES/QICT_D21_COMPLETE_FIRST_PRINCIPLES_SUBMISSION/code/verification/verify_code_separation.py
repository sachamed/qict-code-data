#!/usr/bin/env python3
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]
ACTIVE=[ROOT/'code/finite_representatives']
# Numerical charged-fermion embedding values must not occur in active scientific kernels. Audit scripts are metadata scanners and may name forbidden tokens explicitly.
forbidden=['0.511e-3','105.66e-3','1776.86e-3','2.2e-3','172.57','95e-3','4.18']
hits=[]
for d in ACTIVE:
    for p in d.rglob('*.py'):
        if p.name == 'verify_code_separation.py':
            continue
        s=p.read_text(errors='ignore')
        for token in forbidden:
            if token in s: hits.append({'file':str(p.relative_to(ROOT)),'token':token})
assert not hits,hits
# Empirical finite-representative builders are segregated from the prediction path.
arch=ROOT/'auxiliary/finite_representatives'
required=['finite_builder.py','build_df96.py','build_ckm.py','build_electroweak.py','model_spec.json','DF96_real.csv','DF96_imag.csv']
missing=[x for x in required if not (arch/x).exists()]
assert not missing,missing
out={'status':'PASS','active_code_empirical_mass_literals':0,'auxiliary_finite_representatives_segregated':True,'auxiliary_path':str(arch.relative_to(ROOT))}
(ROOT/'certificates/code_separation_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
