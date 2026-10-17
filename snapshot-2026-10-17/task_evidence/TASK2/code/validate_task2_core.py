#!/usr/bin/env python3
from __future__ import annotations
import json, math, hashlib, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results'
U = ROOT/'upstream'
if not (U/'TASK1_MICROSCOPIC_OPERATOR_CERTIFICATE.json').exists():
    U = ROOT.parent/'TASK1'/'results'/'microscopic_closure'

def load(p):
    with open(p,'r',encoding='utf-8') as f: return json.load(f)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def c(z): return complex(z[0],z[1])

checks={}
notes=[]
# Upstream fingerprint and Task1 status
up=load(U/'TASK1_MICROSCOPIC_OPERATOR_CERTIFICATE.json')
rank=load(R/'TASK2_RANK6_CHIRAL_PARITY_REDUCTION.json')
fp=rank['upstream_task1_operator_fingerprint']
# tolerate certificate schemas that expose fingerprint under different top-level keys by searching values
up_txt=json.dumps(up,sort_keys=True)
checks['task1_fingerprint_bound']=fp in up_txt
checks['task1_fingerprint_expected']=fp=='2e7f0710866ab7d7783ed1ce61b14267494467e73e7f82f9d10aeb134ee5dfe6'

# Revalidate exact 6x6 rank reduction on fixed regression coefficients
abc=[c(z) for z in rank['temporal_worldline_regression_corrected_for_chiral_parity']['weighted_abc']]
a,b,cc=abc
I2=np.eye(2,dtype=complex); C=np.array([[0,1],[1,0]],complex)
I3=np.eye(3,dtype=complex); X=np.roll(np.eye(3,dtype=complex),1,axis=1); Z=np.diag([1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)])
T=(a*np.kron(I2,I3)+b*np.kron(C,X)+cc*np.kron(I2,Z))/3
Dplus=a**3+b**3+cc**3; Dminus=a**3-b**3+cc**3
det_formula=Dplus*Dminus/(27**2)
det_direct=np.linalg.det(T)
sv=np.linalg.svd(T,compute_uv=False)
K=float(-np.sum(np.log(sv))/6)
checks['rank6_det_formula']=abs(det_direct-det_formula)<2e-14
checks['rank6_K_formula']=abs(K-rank['temporal_worldline_regression_corrected_for_chiral_parity']['K_formula'])<5e-14
checks['rank6_positive_leakage']=float(np.min(np.linalg.eigvalsh(np.eye(6)-T.conj().T@T)))>0

# Strong local-link regression consistency
reg=load(R/'TASK2_FULL_LINK_EW_ROUTE_REGRESSION.json')
svr=np.array(reg['singular_values'],float)
Kreg=float(-np.sum(np.log(svr))/6)
checks['full_link_regression_K_recomputed']=abs(Kreg-reg['K_UV_regression'])<5e-14
checks['full_link_regression_positive_leakage']=reg['leakage_min_eig']>0.98

# Error propagation at fine Task1 cutoffs
err=load(R/'TASK2_PARITY_ERROR_BUDGET.json')
fine=err['rows'][-1]
checks['fine_task1_error_bound_finite']=bool(fine['finite_log_bound'])
checks['fine_task1_K_error_below_3p1e8']=fine['K_error_bound']<3.1e-8

# Spatial U1 exact sublayer
sp=load(R/'TASK2_SPATIAL_U1_SUBLAYER_CERTIFICATE.json')
checks['spatial_u1_all_checks']=all(bool(v) for v in sp['checks'].values())

# Temporal route/C10 layer
tmp=load(R/'TASK2_TEMPORAL_ROUTE_C10_CERTIFICATE.json')
checks['temporal_layer_all_checks']=all(bool(v) for v in tmp['checks'].values())
checks['temporal_grid_stable']=tmp['EW_temporal_route_reference']['grid_K_peak_to_peak']<1e-12
checks['temporal_positive_leakage']=tmp['EW_temporal_route_reference']['leakage_min_eig']>0.98

# C10 convergence
c10=load(R/'C10_EXTENDED_R_CONVERGENCE.json')
last=c10['R_sequence'][-1]
cubic=c10['potential_extrapolation_in_1_over_R']['3']
checks['c10_R40_local_flux']=last['successive_restricted_flux_max_abs']<4e-9
checks['c10_cubic_internal_resistance']=cubic['abs_error_to_exact']<1.3e-8

# Canonical compact Kato regulators
for fn,thr in [('Q3_R1_CANONICAL_KATO_DENSE_W4.json',1e-11),('Q3_R1_CANONICAL_KATO_DENSE_W6.json',1e-11),('Q3_R1_CANONICAL_KATO_DENSE_W8.json',1e-11),('Q3_R1_W12_KATO_CONTINUATION.json',2e-12)]:
    d=load(R/fn)
    if 'max_residual' in d: res=d['max_residual']
    else: res=max(float(x['residual']) for x in d.get('rows',[]) if 'residual' in x)
    checks['kato_'+fn.replace('.json','')]=res<thr
# U1 S24 orbital Kato regulator
s24=load(R/'U1_L2_KATO_S24_ORBIT.json')
checks['u1_s24_kato_residual']=s24['selected']['U_residual']<2e-12
checks['u1_s24_overlap']=s24['selected']['overlap_prev']>0.99999
checks['u1_s24_rayleigh_modulus']=abs(s24['selected']['mod_rayleigh']-1.0)<2e-14

# C10 string-lift audit: enforce the physical branch used in package
qa=load(R/'TASK2_C10_QUANTUM_LIFT_AUDIT.json')
checks['c10_quantum_lift_audit']=qa['status']=='PASS_C10_STRING_LIFT_NOT_EQUAL_TANGENT_DRESSING'

# No empirical particle input in principal certificates
principal=[rank,err,reg,sp,tmp,qa]
checks['no_external_particle_data']=all(d.get('external_particle_data_used') is False for d in principal)

checks={k: bool(v) for k,v in checks.items()}
ok=all(checks.values())
cert={
  'status':'PASS_TASK2_EXECUTED_OPERATOR_AND_REGULATOR_CORE' if ok else 'CHECK_TASK2_EXECUTED_OPERATOR_AND_REGULATOR_CORE',
  'checks':checks,
  'upstream_task1_operator_fingerprint':fp,
  'recomputed':{
    'rank6_regression_det_abs_residual':float(abs(det_direct-det_formula)),
    'rank6_regression_K':K,
    'full_link_EW_regression_K':Kreg,
    'fine_task1_propagated_K_bound':fine['K_error_bound'],
    'C10_R40_local_flux_change':last['successive_restricted_flux_max_abs'],
    'C10_cubic_internal_resistance_abs_error':cubic['abs_error_to_exact'],
    'U1_S24_Kato_residual':s24['selected']['U_residual'],
    'U1_S24_overlap_S22':s24['selected']['overlap_prev']
  },
  'claim_boundary':{
    'full_3d_micro_K_UV_promoted':False,
    'reason':'The common 24-link 3D PEPO route scalars h_10,h_9,h_8 have not yet been numerically contracted in one shared U(1)xSU(2)xSU(3)+CAR tensor network. Reduced/link/temporal regressions are therefore not relabeled as K_UV^micro.',
    'exact_promotion_formula':'K_UV=log(3)-[log|a^3+b^3+c^3|+log|a^3-b^3+c^3|]/6, with (a,b,c)=route phases*(h_10,h_9,h_8).'
  }
}
out=R/'TASK2_EXECUTED_CORE_CERTIFICATE.json'
out.write_text(json.dumps(cert,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(cert,indent=2))
if not ok: sys.exit(2)
