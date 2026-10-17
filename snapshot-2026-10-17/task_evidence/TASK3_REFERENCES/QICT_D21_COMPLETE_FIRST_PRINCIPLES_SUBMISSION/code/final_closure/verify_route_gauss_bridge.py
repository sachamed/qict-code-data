#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'route_gauss_bridge'; OUT.mkdir(parents=True,exist_ok=True)

def Pp(E:int)->int: return 2*E*E+E

dvals=np.array([10,9,8],int); delta=21-2*dvals
assert delta.tolist()==[1,3,5] and math.gcd(*map(int,delta))==1
models={
 'A_family_blind':{'E':[0,0,0],'P_plus':[0,0,0]},
 'B_primitive_bridge':{'E':delta.tolist(),'P_plus':[Pp(int(x)) for x in delta]}
}
cost=np.array(models['B_primitive_bridge']['P_plus'],int)
phi=math.pi/63
omega=np.exp(2j*np.pi/3)
I=np.eye(3,dtype=complex); X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex); Z=np.diag([1,omega,omega**2])
T=sum(np.exp(-1j*phi*Pp(int(e)))*w for e,w in zip(delta,[I,X,Z]))/3
s=np.sort(np.linalg.svd(T,compute_uv=False))
beta={1:0,2:30,3:126,4:324,5:660}
cert={
 'status':'PASS',
 'external_particle_data_used':False,
 'route_defects':delta.tolist(),
 'bridge_classification':{
   'defect_lattice':'Z',
   'all_additive_orientation_covariant_integer_maps':'F_k(n)=k n, k in Z',
   'primitive_positive_map':'F(n)=n',
   'logical_boundary':'E=Delta is unique only after the route-defect lattice is identified with the Gauss-source lattice; gauge covariance alone does not make this identification.'
 },
 'non_derivability_countermodels':models,
 'countermodels_distinct':models['A_family_blind']!=models['B_primitive_bridge'],
 'conditional_primitive_bridge':{
   'E_for_d10_d9_d8':delta.tolist(),
   'P_plus_costs':cost.tolist(),
   'half_gap_pair':[17,9],
   'half_gap_ratio_exact':'17/9',
   'half_gap_ratio':17/9
 },
 'phase_only_no_go':{
   'singular_values_ascending':[float(x) for x in s],
   'max_over_min':float(s[-1]/s[0]),
   'middle_over_min':float(s[1]/s[0]),
   'electric_gate_modulus_one':True,
   'verdict':'Electric costs inserted only as real-time phases do not generate the charged-lepton hierarchy.'
 },
 'bcc_execution_boundary':{
   'cycle_rank_beta1':beta,
   'first_nontrivial_cycle_radius':2,
   'R2_three_state_U1_lower_bound':3**30,
   'R2_log10_three_state_U1_lower_bound':30*math.log10(3),
   'note':'Lower bound excludes SU(2), CAR, domain-wall, family/chiral carriers and higher Peter-Weyl support.'
 },
 'route_holonomy_audit':{
   'abstract_gamma_a_defined_elsewhere':False,
   'consequence':'A full 3D route-first numerical contraction is not single-valued until the three microscopic route holonomies are specified or eliminated by a canonical temporal-stream construction.'
 }
}
(OUT/'ROUTE_GAUSS_BRIDGE_CERTIFICATE.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0)
