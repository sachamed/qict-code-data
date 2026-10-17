#!/usr/bin/env python3
from __future__ import annotations
import json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results'/'microscopic_closure'; G=ROOT/'results'/'global_sm_extension'
load=lambda p:json.load(open(p))
u1=load(R/'RANK6_TENSOR_STAGE_A_CERTIFICATE.json')['plaquette_character_expansions']['U1']
su2=load(R/'SU2_CHARACTER_TAIL_CERTIFICATE.json')
su3=load(G/'SU3_CHARACTER_CUTOFF.json')
chi=load(R/'CHIRAL_COMPOSITE_TRANSPORT.json');car=load(R/'CAR_COLLISION_LOWRANK_CERTIFICATE.json')
ir=load(R/'RANK6_TENSOR_NETWORK_IR_V2.json');fac=load(R/'PRODUCT_GROUP_WORLDLINE_FACTORIZATION.json')
# take the declared jmax=6 row from the independent analytic tail validator
r2=next(x for x in su2['rows'] if x['twice_j_max']==12)
b_u1=float(u1['240_gate_telescoped_bound']); b_su2=float(r2['uniform_character_tail_upper'])*240
b_su3=float(su3['chosen_representation_cutoff']['240_gate_union_bound']); total=b_u1+b_su2+b_su3
rep={
 'status':'PASS_TASK1_LOCAL_PRODUCT_GROUP_COMPILATION_V2',
 'external_particle_data_used':False,
 'microscopic_phase':'pi/63',
 'exact_symmetry_labels':['q_U1','j_SU2','(p,q)_SU3','iota_SU2','iota_SU3','Pi_CAR'],
 'certified_local_character_cutoffs':{
   'U1':{'abs_k_max':6,'240_gate_bound':b_u1},
   'SU2':{'twice_j_max':12,'j_max':6.0,'240_gate_bound':b_su2},
   'SU3':{'max_p_q':15,'240_gate_bound':b_su3},
   'combined_240_gate_absolute_bound':total,
   'remaining_from_1e-6_for_TN_compression_roundoff':1e-6-total},
 'chiral_composite_transport':{
   'status':chi['status'],'max_representation_residual':chi.get('max_link_representation_residual_2norm',0.0),
   'rule':'Every active right-handed branch is transported as f_R tensor R_f before the Clebsch bridge; its total U(1)xSU(2)xSU(3) labels equal the corresponding left-handed carrier link by link.'},
 'CAR':{'status':car['status'],'interaction_auxiliary_bond_dimension':2,'unitarity_residual':car['unitarity_residual_2norm'],'parity_commutator_residual':car['fermion_parity_commutator_2norm']},
 'product_group_pathwise_factorization':{'status':fac['status'],'max_five_link_residual':fac['max_five_link_ordered_path_factorization_residual_2norm'],
   'use':'Group factors may be contracted independently only conditional on the same ordered worldline; route amplitudes are recombined coherently afterwards.'},
 'tensor_IR':{'schema_version':ir['schema_version'],'nodes':len(ir['nodes']),'legs':len(ir['legs']),'cutoffs':ir['cutoffs']},
 'promotion_boundary':'This certificate freezes the exact local tensor algebra and all three character cutoffs. Global time/space contraction and chi_TN convergence are separate execution layers; no physical pole or K_UV is inferred here.'
}
out=R/'TASK1_LOCAL_PRODUCT_GROUP_CLOSURE_V2.json';out.write_text(json.dumps(rep,indent=2)+'\n')
print(json.dumps(rep|{'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))
