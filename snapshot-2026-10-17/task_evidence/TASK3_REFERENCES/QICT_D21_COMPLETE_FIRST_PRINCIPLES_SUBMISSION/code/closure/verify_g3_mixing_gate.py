#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
rng=np.random.default_rng(321003)
G=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); G+=2*np.eye(3)
w,V=np.linalg.eigh(G.conj().T@G); U=G@(V*(1/np.sqrt(w)))@V.conj().T
unit=float(np.linalg.norm(U.conj().T@U-np.eye(3)))
physical=ROOT/'data'/'g3_physical_ckm_pmns.json'
g2=ROOT/'data'/'g2_physical_compact_3d_edges.csv'
cert={'observable':'G3 interacting mixing','logical_status':'DEFINED_OBSERVABLE','polar_extraction_unit_test_residual':unit,
      'g2_physical_edge_available':g2.exists(),'physical_mixing_payload_present':physical.exists(),
      'evaluation_condition':'REQUIRES_CONVERGED_G2_EDGES_AND_INTERACTING_WEAK_VERTICES' if not g2.exists() else 'G2_EDGE_INPUT_AVAILABLE',
      'status':'PASS' if unit<1e-11 and not physical.exists() else 'FAIL'}
(CERT/'g3_mixing_gate_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status']=='PASS' else 1)
