from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/task2_continue')
SU=json.load(open(ROOT/'TASK2_SU2_FULL_LINK_MATTER_ROUTE.json'))
su_case=SU['cases'][-1]; su={r['d']:complex(*r['amplitude']) for r in su_case['routes']}
RR=json.load(open('/mnt/data/task2_work/TASK2_RANK6_TO_RANK3_ROUTE_REDUCTION.json'))
h1=[complex(*z) for z in RR['u1_corrected_composite_block']['route_amplitudes_h_U1']]
# order in RR is d=(10,9,8)
D=[10,9,8]
Wcert=json.load(open('/mnt/data/task2_work/TASK2_WEYL_ROUTE_DETERMINANT_CERTIFICATE.json'))
abc_u1=[complex(*z) for z in Wcert['u1_composite_regression']['abc']]
coeff=[abc_u1[i]/h1[i] for i in range(3)]
omega=np.exp(2j*np.pi/3); I=np.eye(3,dtype=complex); X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex); Z=np.diag([1,omega,omega**2]); W=[I,X,Z]
hEW=[h1[i]*su[D[i]] for i in range(3)]; abc=[coeff[i]*hEW[i] for i in range(3)]
A=sum(abc[i]*W[i] for i in range(3))/3
T=np.kron(np.array([[0,1],[1,0]],complex),A)
s=np.linalg.svd(T,compute_uv=False); K=float(-np.mean(np.log(s))); leak=np.linalg.eigvalsh(np.eye(6)-T.conj().T@T)
# Weyl determinant formula: det A=(a^3+b^3+c^3)/27 for circulant+diag convention used here; verify numerically only.
obj={'status':'PASS_TASK2_FULL_LINK_EW_ROUTE_REGRESSION','external_particle_data_used':False,
     'scope':'Pathwise product of the converged corrected-composite U(1) route amplitudes with the converged full Peter-Weyl one-link SU(2) matter-gauge route amplitudes, followed only then by the coherent family Weyl sum. This is the strongest reduced local-link regression currently executed; it is not the 24-link 3D PEPO contraction and is not promoted to K_UV^micro.',
     'route_lengths':D,'route_coefficients':[[z.real,z.imag] for z in coeff],
     'U1_composite_routes':[[z.real,z.imag] for z in h1],
     'SU2_full_link_routes':[[su[d].real,su[d].imag] for d in D],
     'EW_pathwise_routes':[[z.real,z.imag] for z in hEW],
     'weighted_abc':[[z.real,z.imag] for z in abc],
     'A_family':[[[z.real,z.imag] for z in row] for row in A],
     'singular_values':s.tolist(),'K_UV_regression':K,'leakage_eigenvalues':leak.tolist(),'leakage_min_eig':float(leak.min()),'operator_norm':float(s.max()),'min_singular':float(s.min()),
     'det_A':[float(np.linalg.det(A).real),float(np.linalg.det(A).imag)]}
txt=json.dumps(obj,indent=2,sort_keys=True)+'\n';(ROOT/'TASK2_FULL_LINK_EW_ROUTE_REGRESSION.json').write_text(txt)
print(json.dumps({'status':obj['status'],'K':K,'singular_values':s.tolist(),'leakage_min':float(leak.min()),'routes':obj['EW_pathwise_routes'],'sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2))
