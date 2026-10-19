"""Numerical consistency checks for the independent QICT cosmology companion.
These checks do NOT validate QICT's physical dark sector or a CMB likelihood.
"""
import json, math
from pathlib import Path
p=Path(__file__).parent/'results.json'
r=json.loads(p.read_text())
def check(label, v, target, reltol):
    assert math.isfinite(v), (label, v)
    delta=abs(v-target)/max(abs(target),1e-50)
    assert delta<=reltol, (label,v,target,delta)
    print(f'PASS {label}: {v:.12g}; relative discrepancy {delta:.3g}')
check('sound-horizon geometry',r['cosmology']['ell_acoustic'],301.85844,1e-6)
check('SIS Einstein radius',r['lensing']['thetaEinstein_arcsec'],.888390,1e-5)
check('SIS lens time-delay',r['lensing']['time_delay_days'],42.930934,1e-5)
check('QICT-independent Planck reconstruction of Lambda',r['cosmology']['Lambda_m_minus2'],1.090764e-52,1e-5)
check('linear-spectrum sigma8 normalization',r['halo']['sigma8_reproduced'],.811,1e-7)
check('freeze-in integrated / analytic',r['additional']['freezein_relic_numerical_to_analytic_ratio'],1.0,2e-9)
check('Higgs-portal SI ratio at 40 GeV',r['dark_matter_benchmarks']['relative_to_LZ_40GeV_min'],240.0720,1e-5)
check('spherical-collapse threshold',r['spherical_top_hat']['collapse_scale_factor'],0.4614217772,1e-6)
check('Saha-equilibrium redshift',r['saha_recombination']['equilibrium_xe_half_redshift'],1369.006,1e-5)
assert r['dark_matter_benchmarks']['freezein_width_over_H_at_MA'] < 1e-6
print('PASS freeze-in non-equilibrium diagnostic')
print('ALL CHECKS PASSED (computational reproducibility only)')
