#!/usr/bin/env python3
"""Certificate for no-dial C11/G2 mass extraction and absolute-scale identifiability."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CERT=ROOT/'certificates'
EDGE=DATA/'g2_physical_compact_3d_edges.json'
FERMI=CERT/'FERMI_BCC_TAU_STAR_CONSTRAINT.json'
OUT=CERT/'C11_ABSOLUTE_MASS_IDENTIFIABILITY.json'
fermi=json.loads(FERMI.read_text()) if FERMI.exists() else {}
out={
 'status':'PASS_ABSOLUTE_MASS_IDENTIFIABILITY',
 'claim_status':'PROVEN_FORMULA_AND_NO_GO__NUMERICAL_EXTRACTION_REQUIRES_PAYLOADS',
 'formula':{
   'rest_energy':'E_fa(0)=(hbar/tau_*) epsilon_fa_edge',
   'relativistic_mass':'m_fa=(hbar/(c^2 tau_*)) epsilon_fa_edge',
   'same_clock_ratio':'m_fa/m_fb=epsilon_fa_edge/epsilon_fb_edge when both channels share the established relativistic normalization'
 },
 'dimensional_no_go':'No monomial in hbar and c times a dimensionless microscopic invariant has dimensions of time; tau_* is therefore not derivable from the present dimensionless microscopic sector.',
 'g2_edge_payload_present':EDGE.exists(),
 'independent_tau_point_measurement_present':False,
 'fermi_tau_status':fermi.get('taxonomy',{}).get('tau_star','NOT_AVAILABLE'),
 'fermi_use':'constraint only; never used as a point calibration for particle masses',
 'numeric_mass_ratios_status':'COMPUTABLE_NO_DIAL_FROM_G2_EDGES' if EDGE.exists() else 'DEFINED_OBSERVABLE_NO_CONVERGED_G2_EDGE_PAYLOAD',
 'absolute_masses_status':'REQUIRES_INDEPENDENT_TAU_STAR_POINT_MEASUREMENT_AND_G2_EDGES',
 'external_particle_mass_data_used':False
}
if EDGE.exists(): out['edge_payload_sha256']=hashlib.sha256(EDGE.read_bytes()).hexdigest()
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
