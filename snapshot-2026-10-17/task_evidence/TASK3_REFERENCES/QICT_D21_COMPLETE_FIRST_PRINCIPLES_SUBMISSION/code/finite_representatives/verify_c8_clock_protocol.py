#!/usr/bin/env python3
"""Six-clock C8 estimator and uncertainty propagation; no mass calibration."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True); CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
labels=['DeltaT10','DeltaT9','DeltaT8','Tq','Tnu','T126'];n=np.array([1.,3.,5.,14.,21.,126.]);sigma=np.array([.02,.02,.02,.05,.05,.10]);T=n.copy();w=1/sigma**2
hat=float(np.sum(w*n*T)/np.sum(w*n*n));sighat=float(1/math.sqrt(np.sum(w*n*n)));res=(T-n*hat)/sigma
cert={'status':'PASS','external_particle_mass_data_used':False,'clock_multipliers':dict(zip(labels,[1,3,5,14,21,126])),
'exact_relations':['DeltaT9/DeltaT10=3','DeltaT8/DeltaT10=5','Tq/DeltaT10=14','Tnu/DeltaT10=21','T126/DeltaT10=126','Tnu/Tq=3/2'],
'estimator':'tau_hat=[sum_i n_i T_i/sigma_i^2]/[sum_i n_i^2/sigma_i^2]','independent_error':'sigma_tau=[sum_i n_i^2/sigma_i^2]^(-1/2)',
'mass_error_if_epsilon_known':'(sigma_m/m)^2=(sigma_epsilon/epsilon)^2+(sigma_tau/tau)^2 for independent errors',
'synthetic_unit_test_tau':hat,'synthetic_unit_test_sigma':sighat,'synthetic_max_standardized_residual':float(np.max(np.abs(res))),
'physical_tau_value':'NOT_MEASURED_IN_THIS_PACKAGE','note':'The synthetic unit test validates the estimator only. Physical tau_* requires mass-independent receiver timing data.'}
assert abs(hat-1)<1e-14 and np.max(np.abs(res))<1e-12
(CERT/'c8_clock_protocol_certificate.json').write_text(json.dumps(cert,indent=2)+'\n');print(json.dumps(cert,indent=2))
