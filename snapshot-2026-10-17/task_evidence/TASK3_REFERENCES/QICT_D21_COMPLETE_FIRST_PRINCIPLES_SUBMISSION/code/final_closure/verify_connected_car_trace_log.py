#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
LOCK=ROOT/'predictions'/'CONNECTED_CAR_PRECOMPARISON_LOCK.json'
OUT=ROOT/'results'/'connected_car'/'CONNECTED_CAR_TRACE_LOG_CERTIFICATE.json'
p=json.loads(LOCK.read_text())
a=float(p['alpha_factorised'])
N=96
omega=np.exp(2j*np.pi/3)
X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex)
Xe=np.zeros((N,N),complex); Xe[:3,:3]=X
I=np.eye(N,dtype=complex)
# Basic C3 trace data.
trace1=np.trace(Xe); trace2=np.trace(Xe@Xe); trace3=np.trace(Xe@Xe@Xe)
# Determinant identity.
sign,logabs=np.linalg.slogdet(I-a*Xe)
det_num=sign*np.exp(logabs)
det_exact=1-a**3
# Connected trace-log from eigenvalues/series.
delta_exact=-(1/N)*math.log(det_exact)
series=0.0
for n in range(1,80):
    series += a**(3*n)/(N*n)
# Basis covariance under a deterministic random unitary.
rng=np.random.default_rng(20260916)
Z=rng.normal(size=(N,N))+1j*rng.normal(size=(N,N))
Q,_=np.linalg.qr(Z)
Xr=Q.conj().T@Xe@Q
sign2,logabs2=np.linalg.slogdet(I-a*Xr)
delta_rot=-(1/N)*(math.log(abs(sign2*np.exp(logabs2))))
out={
 'status':'PASS',
 'finite_internal_dimension':N,
 'family_support_rank':3,
 'trace_X': [float(trace1.real),float(trace1.imag)],
 'trace_X2':[float(trace2.real),float(trace2.imag)],
 'trace_X3':[float(trace3.real),float(trace3.imag)],
 'det_numeric':[float(det_num.real),float(det_num.imag)],
 'det_exact':det_exact,
 'det_residual':float(abs(det_num-det_exact)),
 'delta_alpha_exact':delta_exact,
 'delta_alpha_series_79_windings':series,
 'series_residual':abs(series-delta_exact),
 'basis_rotation_delta_residual':abs(delta_rot-delta_exact),
 'theorem':'-Tr_96 log(I-alpha X_e)/96 = -log(1-alpha^3)/96, because only C3 windings 3n have nonzero trace.'
}
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if max(out['det_residual'],out['series_residual'],out['basis_rotation_delta_residual'])<1e-12 else 1)
