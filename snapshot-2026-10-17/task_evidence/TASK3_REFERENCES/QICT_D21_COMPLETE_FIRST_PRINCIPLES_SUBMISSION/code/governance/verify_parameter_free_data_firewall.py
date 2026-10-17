#!/usr/bin/env python3
"""Static audit: parameter-free prediction code must not read comparison tables."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'
protected=[ROOT/'code/predictions/mass_spectrum_from_g2_edges.py',ROOT/'code/predictions/mixing_from_interacting_vertices.py',ROOT/'code/closure/verify_c11_family_current_observable.py',ROOT/'code/photon_metrology/verify_bcc_cubic_harmonic_fingerprint.py']
forbidden=['PDG_','FERMI_','experimental_mass','observed_mass','NuFIT','CKM_PDG','PMNS_PDG']
hits=[]
for p in protected:
    t=p.read_text()
    for s in forbidden:
        if s.lower() in t.lower(): hits.append({'file':str(p.relative_to(ROOT)),'token':s})
out={'status':'PASS' if not hits else 'FAIL','claim_status':'COMPUTED','protected_scripts':[str(p.relative_to(ROOT)) for p in protected],'forbidden_tokens':forbidden,'hits':hits,'interpretation':'Mass, C11 family-current, proper-vertex mixing, and BCC shape-prediction scripts are statically separated from experimental comparison tables.'}
(CERT/'PARAMETER_FREE_DATA_FIREWALL.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); raise SystemExit(0 if not hits else 1)
