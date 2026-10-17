#!/usr/bin/env python3
"""Verify the receiver one-clock identities and the registered six-channel estimator.

The numerical timing vector is synthetic and is used only to verify the estimator.
No particle or cosmological observation is read.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True); CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
MULT={'deltaT10':1,'deltaT9':3,'deltaT8':5,'Tq':14,'Tnu':21,'T126':126}
edge=np.array([1.0+0.01*j+0.003*((j*j)%7) for j in range(21)],float)
tau_star=float(edge.mean()); cyclic={}
for d in (10,9,8):
    vals=[]
    for s in range(21):
        idx=[(s+k)%21 for k in range(d)]
        vals.append(edge.sum()-2*edge[idx].sum())
    cyclic[str(d)]=float(abs(np.mean(vals)-(21-2*d)*tau_star))

# Dimensionless synthetic test of the generalized least-squares estimator.
tau_test=2.3456789012345
names=list(MULT); n=np.array([MULT[k] for k in names],float)
sig=np.array([1.0,1.2,1.4,2.0,2.3,3.0])*1e-6
C=np.diag(sig**2); C[0,1]=C[1,0]=0.12*sig[0]*sig[1]; C[3,4]=C[4,3]=0.08*sig[3]*sig[4]
t=n*tau_test; Ci=np.linalg.inv(C)
tau_hat=float((n@Ci@t)/(n@Ci@n)); var=float(1/(n@Ci@n))
mu=0.5851931160827722; a=7.0
E1=mu/tau_test; E2=mu/(a*tau_test); scale_res=float(abs(E2-E1/a)/abs(E2))
rows=list(csv.DictReader(open(DATA/'finite_spectral_invariants.csv')))
schema_ok=bool(rows) and 'spectral_value_over_Etau' in rows[0]
out={
 'status':'PASS',
 'clock_definition':'tau_* is the cyclicly averaged primitive receiver edge time in the reciprocal-path layer.',
 'registered_multipliers':MULT,
 'multiplier_status':{'deltaT10_deltaT9_deltaT8':'reciprocal-path theorem after cyclic averaging',
                      'Tq_Tnu':'declared discrete closure multipliers',
                      'T126':'primitive process period, conditional on a primitive-period witness'},
 'cyclic_average_residuals':cyclic,
 'synthetic_estimator_test':{'tau_input':tau_test,'tau_hat':tau_hat,'absolute_error':abs(tau_hat-tau_test),'variance':var},
 'inverse_scale_covariance_relative_residual':scale_res,
 'finite_spectrum_schema_dimensionless':schema_ok,
 'physical_mass_status':'Not inferred: absolute particle energies require an independently measured receiver clock and the corresponding physical pole calculation.'
}
out['status']='PASS' if max(cyclic.values())<1e-12 and abs(tau_hat-tau_test)<1e-12 and scale_res<1e-14 and schema_ok else 'FAIL'
(CERT/'single_clock_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if out['status']=='PASS' else 1)
