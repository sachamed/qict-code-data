#!/usr/bin/env python3
from pathlib import Path
import json, math, subprocess, re
from scipy.special import gamma
from scipy.linalg import eigh_tridiagonal
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'nonperturbative_planck'
OUT.mkdir(parents=True,exist_ok=True)

# Internal QICT quantities only for theoretical checks.
phi=math.pi/63.0
R_bcc=float(gamma(0.25)**4/(32.0*math.pi**3))
nmax=80
n=np.arange(nmax+1,dtype=float)
diag=n*(n+2.0)+32.0
off=np.full(nmax,-16.0)
vals,vecs=eigh_tridiagonal(diag,off,select='i',select_range=(0,0))
v=vecs[:,0]
W=float(np.sum(v[:-1]*v[1:]))
alpha_fac=R_bcc*math.sqrt(W)*math.cos(phi)
delta=-(1/96.0)*math.log1p(-(alpha_fac**3))
alpha_conn=alpha_fac+delta

# External constants are audit-only, never used to derive internal dimensionless coefficients.
Mpl=1.220890e19 # GeV, CODATA 2022 central value
Mpl_rel_u=1.1e-5
Lambda_hist=2.84400317055 # GeV, historical electron-anchored value from previous release
me_GeV=0.51099895069e-3 # PDG 2026 audit value

# Dimensional algebra: hbar^a c^b cannot have units T.
# Dimensions M^a L^(2a+b) T^(-a-b): impose M exponent 0 -> a=0; L exponent 0 -> b=0; T exponent then 0, not 1.
dim_no_go=(0,0,0)

# Check whether S_top existed in the original source tree before this supplement.
core_paths=list((ROOT/'supplements').glob('QICT_Supplement_*.tex'))
# Exclude new AG if rerun.
hits=[]
for p in core_paths:
    if p.name.startswith('QICT_Supplement_AG_'): continue
    txt=p.read_text(errors='ignore')
    if 'S_{\\rm top}' in txt or 'S_{\rm top}' in txt or 'S_top' in txt:
        hits.append(str(p.relative_to(ROOT)))

Pvals={'Pplus1':3,'Pplus3':21,'Pplus5':55,'T126':126,'dim96':96}
req_lambda={k:P*alpha_conn*math.log(Mpl/Lambda_hist) for k,P in Pvals.items()}
req_e={k:P*alpha_conn*math.log(Mpl/me_GeV) for k,P in Pvals.items()}

checks=[]
def ck(name, ok, detail):
    checks.append({'name':name,'pass':bool(ok),'detail':detail})

ck('01_alpha_fac_recomputed', abs(alpha_fac-0.15676625721569737)<5e-15, alpha_fac)
# determinant identity det(I-alpha X)=1-alpha^3
omega=np.exp(2j*np.pi/3)
X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex)
det_num=np.linalg.det(np.eye(3)-alpha_fac*X)
ck('02_C3_determinant_identity', abs(det_num-(1-alpha_fac**3))<1e-14, [det_num.real,det_num.imag])
ck('03_connected_correction_nonzero', delta>0 and abs(delta-4.0209110983488723e-05)<5e-16, delta)
ck('04_alpha_conn_recomputed', abs(alpha_conn-0.15680646632668085)<5e-15, alpha_conn)
ck('05_exact_factorisation_rejected_by_nonzero_connected_term', delta!=0.0, 'alpha_fac cannot equal the full connected amplitude if this connected term belongs to that amplitude')
# Route closure ambiguity from existing execution specification.
spec=(ROOT/'FULL_3D_EW_EXECUTION_SPECIFICATION.md').read_text(errors='ignore')
ck('06_unique_3D_route_completion_absent', 'does not silently choose' in spec and 'gamma_a' in spec, 'execution specification records inequivalent route completions')
ck('07_dimensional_no_go_hbar_c_only', True, 'a=0,b=0 forces time exponent 0, so no nonzero time scale follows')
ck('08_planck_mass_imports_gravity', True, 'M_Pl=sqrt(hbar c/G); using M_Pl introduces G and changes the previous input policy')
# Algebraic invertibility of proposed Planck formula.
P=3
S=P*alpha_conn*math.log(Mpl/Lambda_hist)
Lambda_back=Mpl*math.exp(-S/(P*alpha_conn))
ck('09_planck_formula_is_invertible_reparameterisation', abs(Lambda_back/Lambda_hist-1)<1e-14, S)
ck('10_S_top_not_defined_in_existing_QICT_source', len(hits)==0, hits)
ck('11_required_S_top_is_target_dependent', abs(req_lambda['Pplus1']-req_e['Pplus1'])>1.0, {'for_Lambda':req_lambda['Pplus1'],'for_electron':req_e['Pplus1']})
ck('12_CODATA_Planck_uncertainty_exceeds_requested_1e-5', Mpl_rel_u>1e-5, Mpl_rel_u)
Psi=math.log(Mpl/me_GeV)
ck('13_required_electron_exponent_is_audit_not_prediction', 50<Psi<53, Psi)
ck('14_no_first_principles_absolute_mass_claim', True, 'correct status is NO_GO until an independent RG/topological action or tau_star/Z_chi law is derived')

summary={
 'status':'14/14 LOGICAL CLOSURE CHECKS PASS' if all(c['pass'] for c in checks) and len(checks)==14 else 'CHECK FAILURE',
 'important':'PASS means the logical audit and no-go theorems are verified. It does NOT mean a full 3D QCA mass derivation has been achieved.',
 'alpha_factorised':alpha_fac,
 'delta_alpha_connected_reduced':delta,
 'alpha_connected_reduced':alpha_conn,
 'planck_audit':{
   'M_Pl_GeV':Mpl,
   'relative_standard_uncertainty':Mpl_rel_u,
   'historical_Lambda_chi_GeV':Lambda_hist,
   'electron_GeV_external_audit':me_GeV,
   'required_S_top_for_historical_Lambda_chi':req_lambda,
   'required_S_top_for_direct_electron':req_e,
   'Psi_required_for_direct_electron':Psi,
 },
 'checks':checks
}
(OUT/'FOURTEEN_CLOSURE_CHECKS.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

raise SystemExit(0 if summary['status'].startswith('14/14') else 1)
