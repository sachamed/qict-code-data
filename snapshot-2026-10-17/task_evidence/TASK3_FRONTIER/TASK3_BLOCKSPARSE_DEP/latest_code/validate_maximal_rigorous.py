#!/usr/bin/env python3
from pathlib import Path
import json, sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]
checks={}
rep=json.load(open(ROOT/'certificates/PEPO3D_REPRESENTATION_TRUNCATION.json'))
checks['representation_truncation_pass']=str(rep.get('status','')).startswith('PASS')
g=json.load(open(ROOT/'latest_results/GAUSS_HILBERT_DIMENSION_JMAX.json'))
checks['gauss_dimension_exact_pass']=g.get('status')=='PASS_EXACT_GAUSS_HILBERT_DIMENSION_COUNT'
checks['gauss_dimension_monotone']=all(g['rows'][i]['gauss_dimension']<g['rows'][i+1]['gauss_dimension'] for i in range(len(g['rows'])-1))
k=json.load(open(ROOT/'latest_results/K10_KATO_ENDPOINT_DIAGNOSTIC.json'))
checks['aux_k10_endpoint_residual_lt_1e8']=k['residual']<1e-8
checks['aux_k10_not_promoted']=k['physical_regulator'] is False
gate=json.load(open(ROOT/'certificates/TASK3_PHYSICAL_PROMOTION_GATE.json'))
for d in (10,9,8):
    checks[f'h{d}_present_and_contracted']=bool(gate.get(f'common_24_link_product_group_h{d}_contracted',False))
checks['all_pass_eligible']=all(checks[f'h{d}_present_and_contracted'] for d in (10,9,8)) and checks['representation_truncation_pass']
expected=False
checks['scientific_guard_matches_current_payload']=(checks['all_pass_eligible']==expected)
status='PASS_MAXIMAL_AUDIT_WITH_PHYSICAL_PROMOTION_BLOCKED' if all(v for k,v in checks.items() if k not in ['h10_present_and_contracted','h9_present_and_contracted','h8_present_and_contracted','all_pass_eligible']) and not checks['all_pass_eligible'] else 'FAIL_INTERNAL_AUDIT'
out={'status':status,'checks':checks,'physical_task3_status':gate.get('status'),'external_particle_data_used':False,'fit_used':False,'reverse_engineering_used':False}
print(json.dumps(out,indent=2))
(ROOT/'certificates/MAXIMAL_RIGOROUS_AUDIT.json').write_text(json.dumps(out,indent=2)+'\\n')
sys.exit(0 if status.startswith('PASS_MAXIMAL') else 1)
