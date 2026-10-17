#!/usr/bin/env python3
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'route_gauss_bridge'; OUT.mkdir(parents=True,exist_ok=True)
# PDG 2026 comparison values, deliberately isolated from the construction script.
me=0.51099895069; mmu=105.6583755; mtau=1776.93
r1=mmu/me; r2=mtau/mmu
rho=math.log(r1)/math.log(r2); qict=17/9
out={
 'status':'POST_CALCULATION_COMPARISON_ONLY',
 'source':'Particle Data Group 2026 charged-lepton summary/listings',
 'masses_MeV':{'electron':me,'muon':mmu,'tau':mtau},
 'ratios':{'mu_over_e':r1,'tau_over_mu':r2,'tau_over_e':mtau/me},
 'log_hierarchy_ratio':rho,
 'conditional_QICT_17_over_9':qict,
 'relative_difference':(rho-qict)/qict,
 'claim_boundary':'The numerical proximity does not promote 17/9 to a mass prediction; the dynamical map from electric costs to the rank-six chiral Schur singular values is not derived by the bridge theorem.'
}
(OUT/'POST_PDG_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
