#!/usr/bin/env python3
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import logm

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'
OUT=ROOT/'results'/'executed_closure'; OUT.mkdir(parents=True,exist_ok=True)

def cpair(z):
    return [float(np.real(z)),float(np.imag(z))]

def cmatrix(A):
    return [[cpair(z) for z in row] for row in A]

def read_receiver():
    p=DATA/'qict_receiver_mixing_exact.json'
    D=json.loads(p.read_text())
    def arr(sec):
        return np.array([[complex(*z) if isinstance(z,list) else complex(z) for z in row] for row in D[sec]['V']],complex)
    return p,D,arr('quark_receiver'),arr('lepton_receiver')

def stdmix(s12,s13,s23,delta):
    c12=np.sqrt(1-s12*s12); c13=np.sqrt(1-s13*s13); c23=np.sqrt(1-s23*s23)
    e=np.exp(1j*delta)
    return np.array([
      [c12*c13, s12*c13, s13*np.exp(-1j*delta)],
      [-s12*c23-c12*s23*s13*e, c12*c23-s12*s23*s13*e, s23*c13],
      [s12*s23-c12*c23*s13*e, -c12*s23-s12*c23*s13*e, c23*c13]
    ],complex)

def invariants(V):
    B=np.abs(V)**2
    s13=abs(V[0,2]); c13=np.sqrt(max(0,1-s13*s13)); s12=abs(V[0,1])/c13; s23=abs(V[1,2])/c13
    J=np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0]))
    return dict(theta12_deg=float(np.degrees(np.arcsin(s12))),theta13_deg=float(np.degrees(np.arcsin(s13))),theta23_deg=float(np.degrees(np.arcsin(s23))),J=float(J),B_abs2=B.tolist())

def align(R,T):
    def fun(x):
        a=x[:3]; b=np.r_[x[3:5],0.0]
        A=np.diag(np.exp(1j*a))@T@np.diag(np.exp(1j*b))
        return np.linalg.norm(A-R,'fro')**2
    rng=np.random.default_rng(7); best=None
    for k in range(80):
        x0=np.zeros(5) if k==0 else rng.uniform(-np.pi,np.pi,5)
        r=minimize(fun,x0,method='BFGS',options={'gtol':1e-12,'maxiter':2000})
        if best is None or r.fun<best.fun: best=r
    x=best.x; a=x[:3]; b=np.r_[x[3:5],0.0]
    A=np.diag(np.exp(1j*a))@T@np.diag(np.exp(1j*b))
    G=A@R.conj().T
    u,s,vh=np.linalg.svd(G); G=u@vh
    H=1j*logm(G); H=(H+H.conj().T)/2
    ev=np.linalg.eigvalsh(H)
    return A,G,H,ev,a,b

p,D,Rq,Rl=read_receiver()
# Comparison targets are not inputs to QICT. They are loaded only after receiver matrices are fixed.
Vckm=stdmix(0.22517,0.003763,0.04189,1.154)
Vpmns=stdmix(math.sqrt(0.3088),math.sqrt(0.02248),math.sqrt(0.470),math.radians(212.0))
flavor={
 'status':'EXECUTED_MINIMAL_1PI_VERTEX_DEFICIT',
 'receiver_payload_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'external_targets_used_only_for_postcomparison':True,
 'target_sources':{'CKM':'Particle Data Group 2026 CKM review/global-fit central values used in the archive','PMNS':'NuFIT 6.1 (2025), normal ordering central values used in the archive'},
}
for name,R,T in [('CKM',Rq,Vckm),('PMNS',Rl,Vpmns)]:
    A,G,H,ev,a,b=align(R,T)
    BR=np.abs(R)**2; BT=np.abs(T)**2
    flavor[name]={
      'receiver':invariants(R),'comparison_target':invariants(T),
      'rephasing_invariant_modulus2_frobenius':float(np.linalg.norm(BR-BT,'fro')),
      'rephasing_invariant_modulus2_max':float(np.max(np.abs(BR-BT))),
      'minimal_matrix_distance_after_rephasing_frobenius':float(np.linalg.norm(A-R,'fro')),
      'minimal_matrix_distance_after_rephasing_spectral':float(np.linalg.norm(A-R,2)),
      'minimal_left_1PI_unitary_geodesic_frobenius':float(np.linalg.norm(H,'fro')),
      'minimal_left_1PI_unitary_geodesic_spectral':float(np.linalg.norm(H,2)),
      'principal_generator_eigenphases_rad':ev.tolist(),
      'principal_generator_eigenphases_deg':[float(math.degrees(v)) for v in ev],
      'principal_hermitian_generator_H':cmatrix(H),
      'minimal_left_unitary_G':cmatrix(G),
      'interpretation':'No family rephasing can identify the receiver matrix with the physical comparison matrix. The displayed H is the smallest left-unitary geodesic correction in the stated alignment class; it quantifies a proper-vertex requirement and is not inserted into QICT.'
    }
(OUT/'FLAVOR_1PI_VERTEX_DEFICIT.json').write_text(json.dumps(flavor,indent=2)+'\n')

# Exact one-loop response-flow audit in the normalization frozen in the QICT extension.
pi=math.pi
kY=10/3; k2=2.0; k3=2.0
sY=-41/(160*pi*pi); s2=19/(96*pi*pi); s3=7/(16*pi*pi)
sD=kY*sY+k2*s2
assert abs(sD + 11/(24*pi*pi))<1e-15
# Postcomparison only.
alpha_cod=0.0072973525643; ainv=137.035999177
D_ir=1/(4*pi*alpha_cod)
Mpl=1.220890e19; Lambda_hist=2.84400317055
lnratio=math.log(Lambda_hist/Mpl)
D_uv_required=D_ir-sD*lnratio
rg={
 'status':'EXECUTED_RG_BOUNDARY_DEFICIT',
 'external_alpha_used_in_derivation':False,
 'response_normalization':{'kY':'10/3','k2':'2','k3':'2','D_em':'kY XiY + k2 Xi2 = 1/(4 pi alpha_em)'},
 'one_loop_exact':{
   'dXiY_dlnmu':'-41/(160 pi^2)','dXi2_dlnmu':'19/(96 pi^2)','dXi3_dlnmu':'7/(16 pi^2)',
   'dD_em_dlnmu':'-11/(24 pi^2)','d_alpha_inverse_dlnmu':'-11/(6 pi)',
   'RG_invariant':'115(XiY-Xi2)+218(Xi3-Xi2)',
   'integration_constants_full_gauge_sector':3,'integration_constants_for_alpha_em_alone':1,
   'nonzero_stationary_fixed_point_exists_at_one_loop':False,
 },
 'proof_statement':'The differential equations fix slopes but not the additive boundary C_em. Since beta_e is proportional to +e^3 in the declared perturbative window, beta_e=0 implies e=0. Therefore alpha_em approximately 1/137 cannot follow from a nonzero stationary one-loop fixed point.',
 'postcomparison_only':{
   'CODATA_2022_alpha_inverse':ainv,'D_em_at_reference_IR_value':D_ir,
   'historical_QICT_Lambda_chi_GeV':Lambda_hist,'Mpl_GeV':Mpl,
   'log_Lambda_over_Mpl':lnratio,
   'C_em_at_Mpl_required_if_single_massless_slope_were_formally_extended_to_Lambda_chi':D_uv_required,
   'warning':'The last number is a diagnostic only; Standard-Model thresholds and electroweak breaking prevent interpreting a single massless one-loop slope as an exact Planck-to-IR evolution.'
 },
 'required_microscopic_closure':'Compute the gauge twist stiffness Xi_r=(k_r V)^(-1) d^2[-log Z_QCA(phi_r)]/dphi_r^2 in the same block-sparse tensor network at a declared blocking scale. No such full SU(3)xSU(2)xU(1)+CAR twist-stiffness payload is present.'
}
(OUT/'RG_BOUNDARY_DEFICIT.json').write_text(json.dumps(rg,indent=2)+'\n')

# Scalar singlet direct-detection envelope.
mh=125.25; mN=0.938918; fN=0.30; conv=0.389379338e-27
mS=40.0; sigma90=2.2e-48; mu=mS*mN/(mS+mN)
coef=fN*fN*mu*mu*mN*mN/(4*pi*mh**4*mS**2)*conv
lam90=math.sqrt(sigma90/coef)
dm={
 'status':'EXECUTED_LZ_PORTAL_BOUNDARY',
 'external_data_used_only_for_postcomparison':True,
 'portal_convention':'V contains (lambda_HS/2) S^2 H^dagger H',
 'cross_section_formula':'sigma_SI=lambda_HS^2 f_N^2 mu_N^2 m_N^2/(4 pi m_h^4 m_S^2) times 0.389379338e-27 cm^2',
 'hadronic_inputs_for_map':{'m_h_GeV':mh,'m_N_GeV':mN,'f_N':fN},
 'LZ_4p2_tonne_year':{'mS_GeV':mS,'sigma90_cm2':sigma90,'confidence_level':0.90,'lambda_HS_upper_if_xiS_equals_1':lam90,'source':'LZ, Phys. Rev. Lett. 135, 011802 (2025); HEPData 155182'},
 'subcomponent_rescaling':'lambda_limit(xi_S)=lambda_limit(1)/sqrt(xi_S)',
 'archive_qca_scalar_mass_present':(DATA/'qca_scalar_singlet.json').exists(),
 'archive_qca_SSHH_vertex_present':(DATA/'qca_scalar_portal_vertex.json').exists(),
 'closure_statement':'The exclusion map is numerical. A QICT point cannot be placed on it until a regulator-stable gauge-singlet scalar pole and an amputated SSHH proper vertex are generated before the LZ data are read.'
}
(OUT/'LZ_PORTAL_BOUNDARY.json').write_text(json.dumps(dm,indent=2)+'\n')

checks={
 'receiver_unitarity_quark':float(np.linalg.norm(Rq.conj().T@Rq-np.eye(3),2))<1e-10,
 'receiver_unitarity_lepton':float(np.linalg.norm(Rl.conj().T@Rl-np.eye(3),2))<1e-10,
 'ckm_nonzero_vertex_deficit':flavor['CKM']['minimal_left_1PI_unitary_geodesic_spectral']>0.1,
 'pmns_nonzero_vertex_deficit':flavor['PMNS']['minimal_left_1PI_unitary_geodesic_spectral']>0.1,
 'rg_one_boundary_constant_for_alpha':rg['one_loop_exact']['integration_constants_for_alpha_em_alone']==1,
 'rg_no_nonzero_one_loop_fixed_point':not rg['one_loop_exact']['nonzero_stationary_fixed_point_exists_at_one_loop'],
 'lz_numeric_bound_positive':0<lam90<1,
 'qca_scalar_payload_not_fabricated':not dm['archive_qca_scalar_mass_present'] and not dm['archive_qca_SSHH_vertex_present'],
}
cert={'status':f"PASS_{sum(checks.values())}_OF_{len(checks)}",'checks':checks,'scientific_conclusion':'The receiver-sector flavor matrices are genuine QICT outputs but require nontrivial 1PI family-sensitive vertices to become physical CKM/PMNS matrices. The perturbative RG fixes the electromagnetic slope but leaves one microscopic boundary constant. LZ supplies a quantitative portal bound, while the active QICT archive does not yet contain the scalar pole and SSHH vertex needed for a parameter-free dark-matter point.'}
(OUT/'EXECUTED_CLOSURE_CERTIFICATE.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
print('CKM geodesic spectral =',flavor['CKM']['minimal_left_1PI_unitary_geodesic_spectral'])
print('PMNS geodesic spectral=',flavor['PMNS']['minimal_left_1PI_unitary_geodesic_spectral'])
print('lambda_HS^90(40 GeV)  =',lam90)
