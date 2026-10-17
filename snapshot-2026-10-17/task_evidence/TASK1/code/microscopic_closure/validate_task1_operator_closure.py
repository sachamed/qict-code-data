#!/usr/bin/env python3
from __future__ import annotations
import json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results'/'microscopic_closure'; G=ROOT/'results'/'global_sm_extension'

def J(p): return json.load(open(p))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={
 'route':R/'TEMPORAL_SEMIGROUP_ROUTE_UNIQUENESS.json',
 'carrier':R/'CHIRAL_COMPOSITE_TRANSPORT.json',
 'car':R/'CAR_COLLISION_LOWRANK_CERTIFICATE.json',
 'car1':R/'CAR_ONE_PARTICLE_REDUCTION.json',
 'pepo':R/'TASK1_PEPO_L_SEQUENCE.json',
 'norm':R/'TASK1_OPERATOR_NORM_FREEZE.json',
 'dress':R/'TASK1_R_DRESSING_CONVERGENCE.json',
 'product':R/'PRODUCT_GROUP_WORLDLINE_FACTORIZATION.json',
 'u1':R/'L2_U1_EXACT_CHARACTER_CONTRACTION.json',
 'su2tail':R/'SU2_CHARACTER_TAIL_CERTIFICATE.json',
 'su3':G/'SU3_CHARACTER_CUTOFF.json',
 'sm':G/'FULL_SM_GAUGE_GROUP_CERTIFICATE.json',
 'local':R/'TASK1_LOCAL_PRODUCT_GROUP_CLOSURE_V2.json',
}
d={k:J(v) for k,v in files.items()}
checks={
 'canonical_temporal_routes_unique': d['route']['status'].startswith('PASS') and d['route']['max_semigroup_residual_2norm']<1e-12,
 'composite_chiral_transport_exact': d['carrier']['status'].startswith('PASS') and d['carrier']['max_link_representation_residual']==0.0,
 'car_collision_exact_unitary': d['car']['status'].startswith('PASS') and d['car']['unitarity_residual_2norm']<1e-12,
 'car_one_particle_reduction_exact': d['car1']['status'].startswith('PASS') and d['car1']['finite_sanity_check']['one_particle_block_error']==0.0,
 'full_sm_group_certified': d['sm']['status']=='PASS_8_OF_8',
 'same_pepo_fuses_gauge_matter_car_all_L': d['pepo']['status'].startswith('PASS') and all(x['all_link_nodes_fuse_gauge_matter_car'] for x in d['pepo']['volumes']),
 'exact_symmetry_labels_retained': d['pepo']['exact_labels']==['q_U1','j_SU2','(p,q)_SU3','iota_SU2','iota_SU3','Pi_CAR'],
 'pathwise_product_group_identity': d['product']['status'].startswith('PASS') and d['product']['max_five_link_ordered_path_factorization_residual_2norm']<1e-12,
 'operator_norm_freeze_L2_below_1e-6': d['norm']['status'].startswith('PASS') and d['norm']['freeze_cutoff_for_L2_n1']['operator_norm_bound']<1e-6,
 'adaptive_L2_L12_all_below_1e-6': all(x['one_step_operator_norm_bound']<1e-6 for x in d['norm']['adaptive_volume_sequence_below_1e-6']),
 'chi_sequence_monotone': all(a['chi_TN_exact_block_channels_including_CAR_parity']<b['chi_TN_exact_block_channels_including_CAR_parity'] for a,b in zip(d['norm']['cutoff_and_chi_sequence'],d['norm']['cutoff_and_chi_sequence'][1:])),
 'R_dressing_sequence_pass': d['dress']['status'].startswith('PASS') and d['dress']['exact_flow_checks_max_R12']['max_abs_sink_probability_minus_one']<1e-12,
 'U1_exact_character_contraction_present': d['u1']['status'].startswith('PASS'),
 'SU2_analytic_character_tail_present': d['su2tail']['status'].startswith('PASS'),
 'SU3_deterministic_character_cutoff_present': d['su3']['status'].startswith('EXECUTED') and d['su3']['chosen_representation_cutoff']['240_gate_union_bound']<2.5e-7,
 'no_external_particle_data_in_task1_certificates': all(not x.get('external_particle_data_used',False) for x in d.values() if isinstance(x,dict)),
}
status='PASS_TASK1_MICROSCOPIC_OPERATOR_AND_3D_PEPO_CLOSURE' if all(checks.values()) else 'FAIL_TASK1_OPERATOR_CLOSURE'
frozen={
 'operator':'U_micro^rf = U_int^rf exp(-i phi_* K_B) exp(-i phi_* K_E) U_cell Gamma(S_x) Gamma(S_y) Gamma(S_z)',
 'phi_star':'pi/63','route_completion':'P_d=U_C8^d, d in {10,9,8}',
 'bulk_frozen_cutoff_L2':d['norm']['freeze_cutoff_for_L2_n1'],
 'exact_labels':d['pepo']['exact_labels'],
 'operator_fingerprint_payload':{
   'order':d['pepo']['operator_order'],'routes':d['pepo']['routes'],'labels':d['pepo']['exact_labels'],
   'cutoff':d['norm']['freeze_cutoff_for_L2_n1']}
}
frozen['operator_fingerprint_sha256']=hashlib.sha256(json.dumps(frozen['operator_fingerprint_payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
rep={'status':status,'pass_definition':'Task 1 is closed at the operator level: the complete local compact-group/Haar/CAR contraction is compiled into one symmetry-blocked 3D PEPO. The PEPO input/output and virtual representation/intertwiner legs remain open because they are the microscopic transfer operator itself. No Kato-state expectation, Feshbach projection, pole extraction or mass observable is taken in Task 1.',
 'checks':checks,'passed':sum(checks.values()),'total':len(checks),'frozen_operator':frozen,
 'regulators':{'L':'compiled L=2..6; local quasi-local limit plus adaptive operator-norm table L=2..12','U1_abs_k':'2..7','SU2_twice_j':'4..14','SU3_max_p_q':'7..17','R':'2,3,4,5,6,8,10,12','chi_TN_exact_block_channels':'3200,9800,23328,47432,86528,145800'},
 'crosschecks':{'U1_exact_character_status':d['u1']['status'],'SU2_order8_deterministic_crosscheck_file':'L2_SU2_MATTER_J1_ORDER_8_VERTEX.json','SU3_Weyl_quadrature_max_change':d['su3']['deterministic_Weyl_quadrature_crosscheck']['max_coefficient_change_pq_le_10']},
 'source_sha256':{k:sha(p) for k,p in files.items()}}
out=R/'TASK1_MICROSCOPIC_OPERATOR_CERTIFICATE.json';txt=json.dumps(rep,indent=2,sort_keys=True)+'\n';out.write_text(txt)
print(json.dumps(rep|{'certificate_sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2))
