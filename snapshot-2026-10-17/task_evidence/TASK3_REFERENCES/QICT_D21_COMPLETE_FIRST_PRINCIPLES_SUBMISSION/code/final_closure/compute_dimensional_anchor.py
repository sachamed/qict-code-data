#!/usr/bin/env python3
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parents[2]
LOCK=ROOT/'predictions'/'CONNECTED_CAR_PRECOMPARISON_LOCK.json'
OUT=ROOT/'results'/'connected_car'/'DIMENSIONAL_ANCHOR_ELECTRON.json'
p=json.loads(LOCK.read_text())
a=float(p['alpha_connected'])
me=0.51099895069 # external dimensional anchor, MeV
P_e,P_mu,P_tau=55,21,3
Lambda_chi_MeV=me*math.exp(a*P_e)
mu=Lambda_chi_MeV*math.exp(-a*P_mu)
tau=Lambda_chi_MeV*math.exp(-a*P_tau)
# Conditional photon/LIV consistency map using the LHAASO quadratic lower limit:
Epl_GeV=1.22089e19
EQG2_lower_GeV=6e-8*Epl_GeV
bmin=3/32
Etau_lower_GeV=EQG2_lower_GeV*math.sqrt(2*bmin/3)
hbar_GeVs=6.582119569e-25
taustar_upper_s=hbar_GeVs/Etau_lower_GeV
Zchi_upper=(Lambda_chi_MeV/1000.0)/Etau_lower_GeV
out={
 'status':'ONE_DIMENSIONAL_ANCHOR_CALIBRATION',
 'anchor':{'electron_mass_MeV':me},
 'family_shape':{'alpha_connected':a,'costs':{'e':P_e,'mu':P_mu,'tau':P_tau}},
 'anchored_common_chiral_scale_MeV':Lambda_chi_MeV,
 'anchored_predictions_MeV':{'muon':mu,'tau':tau},
 'scale_factorization':{
   'relation':'Lambda_chi = Z_chi * E_tau, E_tau=hbar/tau_*',
   'important':'m_e fixes Lambda_chi, not tau_* separately. A point measurement of tau_* would determine Z_chi.'
 },
 'conditional_LHAASO_consistency':{
   'E_Pl_GeV_used':Epl_GeV,
   'published_quadratic_limit':'E_QG,2 > 6e-8 E_Pl (95% CL)',
   'QICT_b_min':'3/32',
   'mapping':'E_tau > E_QG,2 sqrt(2 b_min/3)',
   'E_tau_lower_GeV':Etau_lower_GeV,
   'tau_star_upper_s':taustar_upper_s,
   'Z_chi_upper_from_e_anchor':Zchi_upper
 },
 'incorrect_identification_warning':(
   'Setting Z_chi=1 would identify Lambda_chi with E_tau. That is not implied by QICT '
   'and is incompatible with the quoted photon bound; the protected chiral zero requires a small common residue.'
 )
}
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
