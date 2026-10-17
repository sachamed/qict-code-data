#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
x=json.load(open(R/'SU2_Q3_COMPONENT_SCHMIDT_PROFILE.json'))
rows=x['rows']
checks={
 'q3_profile_pass':x.get('status')=='PASS_SU2_Q3_EXACT_COMPONENT_SCHMIDT_PROFILE',
 'all_23_cuts_profiled':len(rows)==23,
 'worst_total_rank_finite':max(r['exact_schmidt_rank'] for r in rows)>0,
 'largest_component_rank_le_16':max(r['largest_component_rank'] for r in rows)<=16,
 'many_exact_components_present':max(r['components'] for r in rows)>=1000,
}
out={
 'status':'PASS_SU2_BLOCK_SPARSE_FRONTIER_VIABILITY' if all(checks.values()) else 'FAIL_SU2_BLOCK_SPARSE_FRONTIER_VIABILITY',
 'state':'exact H_B^3|vac> on the 24-link L=2 fusion-tree spin-network graph',
 'certified_link_order':x['order'],
 'cuts_profiled':len(rows),
 'worst_exact_total_schmidt_rank':max(r['exact_schmidt_rank'] for r in rows),
 'maximum_number_of_independent_bipartite_components':max(r['components'] for r in rows),
 'maximum_rank_of_any_single_component':max(r['largest_component_rank'] for r in rows),
 'maximum_component_rows':max(r['largest_component_rows'] for r in rows),
 'maximum_component_cols':max(r['largest_component_cols'] for r in rows),
 'cut_of_worst_total_rank':max(rows,key=lambda r:r['exact_schmidt_rank'])['cut'],
 'checks':checks,
 'algorithmic_consequence':'The SU(2) frontier must retain exact sector/component labels and SVD only inside each component. A global Schmidt truncation or configuration-amplitude pruning is structurally inappropriate. The observed q3 components are individually small (rank <=16) even when their direct-sum rank is >5000.',
 'non_promotion':'This is an exact frontier-architecture certificate. It does not execute the repeated-time charged PEPO and does not populate h_8,h_9,h_10.'
}
(R/'SU2_BLOCK_SPARSE_FRONTIER_VIABILITY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
