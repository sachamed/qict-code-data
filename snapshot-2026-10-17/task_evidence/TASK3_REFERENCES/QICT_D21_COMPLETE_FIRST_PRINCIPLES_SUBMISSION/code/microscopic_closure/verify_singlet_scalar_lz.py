#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'global_sm_extension'; OUT.mkdir(parents=True,exist_ok=True)
PAY=ROOT/'data'/'qca_scalar_singlet.json'
# External comparison constants only; they are not used to define a QCA point.
mh=125.25 # GeV, comparison convention
mN=0.938918 # isoscalar nucleon GeV
fN=0.30
gev2_to_cm2=0.389379338e-27

def sigma_si(mS,lam):
    mu=mS*mN/(mS+mN)
    sig_gev2=lam**2*fN**2*mu**2*mN**2/(4*math.pi*mh**4*mS**2)
    return sig_gev2*gev2_to_cm2

obj={
 'status':'NO_QCA_SCALAR_POLE_OR_PORTAL_VERTEX_PAYLOAD',
 'claim_status':'DIRECT_DETECTION_MAP_DEFINED__PARAMETER_FREE_POINT_NOT_DERIVED_IN_CURRENT_C1_C11_ARCHITECTURE',
 'qca_payload_present':PAY.exists(),
 'required_qca_observables':['scalar singlet pole mass m_S from a gauge-singlet two-point function','amputated zero-momentum SSHH proper vertex lambda_HS','thermal history or a microscopic production calculation for the relic fraction'],
 'portal_convention':'V contains (lambda_HS/2) S^2 H^dagger H',
 'spin_independent_formula_cm2':'sigma_SI=lambda_HS^2 f_N^2 mu_N^2 m_N^2/(4 pi m_h^4 m_S^2) times (GeV^-2 to cm^2)',
 'LZ_reference':{
   'ordinary_SI_dataset':'LZ 4.2 tonne-year WIMP search, PRL 135, 011802 (2025), HEPData 155182',
   'strongest_reported_90CL_cm2':2.2e-48,
   'mass_at_strongest_limit_GeV':40.0,
   '2026_extended_recoil_search':'2.84 tonne-year high-recoil EFT/inelastic search; one 248 keV nuclear-recoil-like event; global background-only tension 2.6 sigma. This is not a replacement for the ordinary elastic-SI exclusion curve.'
 },
 'external_data_used_only_for_comparison':True,
}
if PAY.exists():
    p=json.loads(PAY.read_text()); mS=float(p['mS_GeV']); lam=float(p['lambda_HS']); frac=float(p.get('relic_fraction',1.0))
    sig=sigma_si(mS,lam); obj['qca_prediction']={'mS_GeV':mS,'lambda_HS':lam,'relic_fraction':frac,'sigma_SI_cm2':sig,'rescaled_sigma_cm2':frac*sig}
    obj['status']='QCA_POINT_COMPUTED_REQUIRES_MASS_DEPENDENT_LZ_CURVE_FOR_FINAL_EXCLUSION'
(OUT/'SINGLET_SCALAR_LZ_CERTIFICATE.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
