#!/usr/bin/env python3
import json, numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent
z=np.load(R/'payload'/'C10_KATO_BOUNDARY_TO_CORE_COMPACT_SECTOR_MAP.npz')
B=z['incidence']; P=z['plaquette_matrix']; Qp=z['plaquette_basis']; Qh=z['harmonic_basis']
t=z['target_divergence_unit']; e=z['affine_representative_unit']; e3=z['affine_representative_qminus3']
checks={
 'unit_divergence':float(np.max(np.abs(B@e-t))),
 'qminus3_divergence':float(np.max(np.abs(B@e3+3*t))),
 'plaquette_cycle':float(np.max(np.abs(B@P))),
 'affine_harmonic_norm':float(np.linalg.norm(Qh.T@e)),
 'loop_harmonic_orthogonality':float(np.linalg.norm(Qp.T@Qh)),
}
ok=checks['unit_divergence']<1e-12 and checks['qminus3_divergence']<1e-12 and checks['plaquette_cycle']<1e-12 and checks['affine_harmonic_norm']<1e-12 and checks['loop_harmonic_orthogonality']<1e-12
print(json.dumps({'status':'PASS' if ok else 'FAIL','checks':checks},indent=2))
raise SystemExit(0 if ok else 1)
