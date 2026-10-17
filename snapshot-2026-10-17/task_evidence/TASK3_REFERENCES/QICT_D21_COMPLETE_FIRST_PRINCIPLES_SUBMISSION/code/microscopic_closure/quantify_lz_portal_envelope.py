#!/usr/bin/env python3
"""Translate the LZ 40-GeV SI benchmark into a Higgs-portal coupling envelope."""
from __future__ import annotations
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'global_sm_extension'/'LZ_PORTAL_ENVELOPE_QUANTITATIVE.json'; OUT.parent.mkdir(parents=True,exist_ok=True)
mh=125.25; mN=0.938918; fN=0.30; conv=0.389379338e-27
mS=40.0; sigma90=2.2e-48
mu=mS*mN/(mS+mN)
coef=fN*fN*mu*mu*mN*mN/(4*math.pi*mh**4*mS**2)*conv
lam90=math.sqrt(sigma90/coef)
obj={
 'status':'EXECUTED_LZ_PORTAL_ENVELOPE__NO_QCA_SCALAR_POINT',
 'portal_convention':'V contains (lambda_HS/2) S^2 H^dagger H',
 'sigma_formula':'sigma_SI=lambda_HS^2 f_N^2 mu_N^2 m_N^2/(4*pi*m_h^4*m_S^2) * 0.389379338e-27 cm^2',
 'comparison_conventions':{'m_h_GeV':mh,'m_N_GeV':mN,'f_N':fN},
 'LZ_4p2_tonne_year_benchmark':{
   'mS_GeV':mS,'sigma90_cm2':sigma90,'CL':0.90,
   'lambda_HS_upper_for_unit_relic_fraction':lam90,
   'subcomponent_scaling':'lambda_HS^90(xi_S)=lambda_HS^90(xi_S=1)/sqrt(xi_S)',
   'source':'LZ Collaboration, Phys. Rev. Lett. 135, 011802 (2025): strongest SI exclusion 2.2e-48 cm^2 at 40 GeV/c^2.'
 },
 'archive_scalar_payload_present':(ROOT/'data'/'qca_scalar_singlet.json').exists(),
 'archive_search_result':'No regulator-stable gauge-singlet scalar pole and no amputated SSHH proper-vertex payload are present in the active C1-C11 archive.',
 'decision':'The LZ map is quantitative, but a parameter-free QICT cross section cannot be calculated until both m_S and lambda_HS are produced by QCA correlators before reading the exclusion curve.'
}
OUT.write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
