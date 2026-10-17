#!/usr/bin/env python3
"""G2 protocol certificate. Validates the convergence gate on synthetic data only."""
from pathlib import Path
import json, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
# Synthetic sequence with known edge, used only to unit-test the extrapolation gate.
Einf=0.43125; a=0.7; b=-0.2
Ls=np.array([16,24,32,48,64],float)
y=Einf+a/Ls**2+b/Ls**3
X=np.c_[np.ones_like(Ls),1/Ls**2,1/Ls**3]
coef=np.linalg.lstsq(X,y,rcond=None)[0]; pred=float(coef[0]); err=abs(pred-Einf)
# Physical data are deliberately absent.
physical_edge_path=ROOT/'data'/'g2_physical_compact_3d_edges.csv'
cert={'gap':'G2','taxonomy':'PROTOCOLIZED','synthetic_gate_test':{'Ls':Ls.astype(int).tolist(),'known_limit':Einf,'recovered_limit':pred,'abs_error':err},
      'physical_compact_edge_dataset_present':physical_edge_path.exists(),
      'required_acceptance':['fixed doubler treatment/class extension selected before run','at least five volumes','joint finite-volume/harmonic/dressing/moment convergence','independent repeat implementation','PHA-2 lock before comparison'],
      'status':'PASS_PROTOCOL_ONLY' if err<1e-12 and not physical_edge_path.exists() else 'FAIL'}
(CERT/'g2_convergence_protocol_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status'].startswith('PASS') else 1)
