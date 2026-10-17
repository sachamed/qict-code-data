#!/usr/bin/env python3
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
# Unit-test only: ratio and independent-error propagation for synthetic edges.
e1,s1=0.42,0.003; e2,s2=0.17,0.002
r=e1/e2; sr=abs(r)*math.sqrt((s1/e1)**2+(s2/e2)**2)
physical=ROOT/'data'/'g4_physical_mass_ratios.csv'; edge=ROOT/'data'/'g2_physical_compact_3d_edges.csv'
cert={'observable':'G4 mass-ratio extraction','logical_status':'DEFINED_OBSERVABLE','synthetic_ratio_unit_test':{'ratio':r,'sigma':sr},
      'physical_edges_present':edge.exists(),'physical_mass_ratio_payload_present':physical.exists(),
      'required_discretization_variants':3,'status':'PASS' if (r>0 and sr>0 and not physical.exists()) else 'FAIL'}
(CERT/'g4_mass_ratio_gate_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status']=='PASS' else 1)
