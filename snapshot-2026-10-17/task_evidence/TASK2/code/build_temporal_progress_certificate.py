import json,hashlib,math
from pathlib import Path
import numpy as np
BASE=Path('/mnt/data/task2_continue')
T1=json.load(open('/mnt/data/task2_continue/task1/QICT_D21_TASK1_MICROSCOPIC_OPERATOR_CLOSURE_PASS_CLEAN/results/microscopic_closure/TASK1_MICROSCOPIC_OPERATOR_CERTIFICATE.json'))
Ns=[13,15,17,19,21]
u1=[json.load(open(BASE/f'temporal/N{N}.json')) for N in Ns]
su=[json.load(open(BASE/f'su2rot/n{n}.json')) for n in [20,40,60,80]]
ew=[json.load(open(BASE/f'ewdiag/N{N}_n{n}.json')) for N in [13,17,21] for n in [20,40,80]]
c10=json.load(open(BASE/'C10_EXTENDED_R_CONVERGENCE.json'))
K_u=np.array([x['K_UV'] for x in u1]); K_e=np.array([x['K_UV'] for x in ew])
ref=next(x for x in ew if x['N_U1']==21 and x['SU2_nmax']==80)
s=np.array(ref['singular_values'])
eps=T1['frozen_operator']['bulk_frozen_cutoff_L2']['operator_norm_bound']
err=float(np.mean(-np.log1p(-eps/s)))
# route amplitude convergence SU2
z=lambda a: np.array([complex(*q) for q in a['route_amplitudes']])
su_ref=z(su[-1]); su_max=max(float(np.max(np.abs(z(x)-su_ref))) for x in su[:-1])
# U1 Kato refinement 11 vs17 if available
u11=json.load(open(BASE/'temporal/N21_s11.json'))
u17=json.load(open(BASE/'temporal/N21_s17.json'))
checks={
 'task1_fingerprint_exact':T1['frozen_operator']['operator_fingerprint_sha256']=='2e7f0710866ab7d7783ed1ce61b14267494467e73e7f82f9d10aeb134ee5dfe6',
 'u1_temporal_N_converged':float(np.ptp(K_u))<1e-12,
 'u1_kato_refinement_converged':abs(u11['K_UV']-u17['K_UV'])<1e-12,
 'su2_temporal_rotor_cutoff_converged':su_max<1e-12,
 'ew_temporal_route_grid_converged':float(np.ptp(K_e))<1e-12,
 'positive_leakage_margin':ref['leakage_min_eig']>1e-3,
 'c10_local_flux_R40_below_1e-8':c10['R_sequence'][-1]['successive_restricted_flux_max_abs']<1e-8,
 'c10_cubic_extrapolation_matches_internal_bcc_resistance':c10['potential_extrapolation_in_1_over_R']['3']['abs_error_to_exact']<1e-7,
}
obj={
 'status':'PASS_TASK2_TEMPORAL_ROUTE_AND_C10_REGRESSION_LAYER' if all(checks.values()) else 'FAIL',
 'scope':'Deterministic regression layer for Task 2: canonical temporal routes d=(10,9,8), composite q=-3 U(1) Kato strip, SU(2) class-function fundamental rotor, route-conditioned product-group recombination, and extended C10 source convergence. This certificate does not identify the local-rotor replacement with the full spatial PEPO contraction.',
 'external_particle_data_used':False,
 'task1_operator_fingerprint_sha256':T1['frozen_operator']['operator_fingerprint_sha256'],
 'U1_temporal':{'N_sequence':Ns,'K_values':K_u.tolist(),'K_peak_to_peak':float(np.ptp(K_u)),'N21_steps11_to17':abs(u11['K_UV']-u17['K_UV']),'route_amplitudes_N21':u1[-1]['route_amplitudes']},
 'SU2_temporal_rotor':{'nmax_sequence':[20,40,60,80],'max_route_amplitude_change_to_n80':su_max,'route_amplitudes_n80':su[-1]['route_amplitudes'],'tail_last_two_n80':su[-1]['kato']['tail_last_two']},
 'EW_temporal_route_reference':{'N_U1':21,'SU2_nmax':80,'K_UV':ref['K_UV'],'singular_values':ref['singular_values'],'leakage_min_eig':ref['leakage_min_eig'],'route_amplitudes':ref['EW_route_amplitudes'],'grid_K_peak_to_peak':float(np.ptp(K_e))},
 'task1_operator_error_propagation_if_same_compression':{'epsilon':eps,'logdet_absolute_bound':err},
 'C10_extended':c10,
 'checks':checks,
}
txt=json.dumps(obj,indent=2,sort_keys=True)+'\n'; p=BASE/'TASK2_TEMPORAL_ROUTE_C10_CERTIFICATE.json';p.write_text(txt);print(json.dumps({'status':obj['status'],'K_EW':ref['K_UV'],'K_grid_ptp':float(np.ptp(K_e)),'U1_ptp':float(np.ptp(K_u)),'SU2_route_max_change':su_max,'C10_R40_flux_change':c10['R_sequence'][-1]['successive_restricted_flux_max_abs'],'task1_log_bound':err,'sha256':hashlib.sha256(txt.encode()).hexdigest(),'checks':checks},indent=2))
