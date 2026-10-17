#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rank6_tensor_local_blocks import self_test

cert=json.loads((ROOT/'results'/'microscopic_closure'/'RANK6_TENSOR_STAGE_A_CERTIFICATE.json').read_text())
ir=json.loads((ROOT/'results'/'microscopic_closure'/'RANK6_TENSOR_NETWORK_IR.json').read_text())
local=self_test()
checks={
 'geometry_8_24_24': cert['geometry']['vertices']==8 and cert['geometry']['links']==24 and cert['geometry']['plaquettes']==24,
 'cellular_b1_3': cert['geometry']['cellular_b1']==3,
 'u1_cutoff_6': cert['plaquette_character_expansions']['U1']['shift_cutoff_abs_k']==6,
 'su2_jmax_6': cert['plaquette_character_expansions']['SU2']['twice_j_max']==12,
 'analytic_error_lt_5e-8': float(cert['plaquette_character_expansions']['combined_analytic_gate_error_bound'])<5e-8,
 'remaining_budget_gt_9e-7': float(cert['plaquette_character_expansions']['remaining_absolute_error_budget_for_TRG_and_roundoff'])>9e-7,
 'pure_intertwiner_mult_13': cert['local_link_tensor']['max_pure_gauge_intertwiner_multiplicity']==13,
 'left_intertwiner_mult_24': cert['local_link_tensor']['max_left_matter_intertwiner_multiplicity']==24,
 'car_one_particle_exact': 'Exact restriction' in cert['CAR']['vacuum_two_point_reduction'],
 'ir_nonfactorized': 'No Z_gauge*T_matter split' in ir['nonfactorization'],
 'ir_nodes_762': len(ir['nodes'])==762,
 'local_block_tests': local['status']=='PASS_LOCAL_BLOCKS',
}
status='PASS' if all(checks.values()) else 'FAIL'
out={'status':status,'checks':checks,'passed':sum(checks.values()),'total':len(checks),
     'claim_boundary':'This certificate validates Step A local tensors, cutoffs and network compilation. It does not certify a completed global rank-six contraction or a numerical K_UV.'}
path=ROOT/'results'/'microscopic_closure'/'RANK6_TENSOR_STAGE_A_VERIFICATION.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
if status!='PASS': raise SystemExit(1)
