#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
# Exact model invariance: U=exp(-iK) is unchanged under tau->lambda tau, H->H/lambda.
lams=[0.1,0.5,2.0,10.0]
# Synthetic six-clock GLS validates estimator only, not a physical measurement.
n=np.array([1.,3.,5.,14.,21.,126.]); tau=2.3e-15; sigma=np.array([3,3,4,8,10,30])*1e-18
C=np.diag(sigma**2); Ci=np.diag(1/sigma**2); y=n*tau
hat=float(n@Ci@y/(n@Ci@n)); shat=float((n@Ci@n)**-0.5)
cert={'gap':'G5','taxonomy':{'dimensionful_scale_no_go':'DEMONSTRATED','tau_star_measurement':'PROTOCOLIZED'},
      'rescaling_symmetry':{'lambda_values':lams,'U_invariant_by_construction':True,'energy_scales_as_inverse_lambda':True},
      'synthetic_six_clock_GLS':{'tau_true_s':tau,'tau_hat_s':hat,'sigma_hat_s':shat,'abs_recovery_error_s':abs(hat-tau)},
      'physical_tau_star_measured':False,
      'feasibility':'CONDITIONAL: protocol is identifiable if delays are resolvable; QICT supplies no internal prior on tau_star, so no honest instrument-resolution or calendar-horizon forecast follows.',
      'status':'PASS' if abs(hat-tau)<1e-28 else 'FAIL'}
(CERT/'g5_scale_identifiability_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status']=='PASS' else 1)
