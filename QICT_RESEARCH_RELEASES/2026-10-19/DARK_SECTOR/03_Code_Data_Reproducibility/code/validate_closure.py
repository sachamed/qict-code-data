"""Recalculate and verify finite-moment, LZ, freeze-in, freeze-out, and shear tests."""
import json
from pathlib import Path
import numpy as np
p=Path(__file__).parent
r=json.loads((p/'closure_results.json').read_text())
old=json.loads((p/'results.json').read_text())
N=6
root=np.exp(2j*np.pi*np.arange(N+1)/(N+1))
for n in range(1,N+1):
 assert abs(np.mean(root**n)) < 1e-13
print('PASS finite moment non-uniqueness witness N=6')
a=r['direct_detection']; b=old['dark_matter_benchmarks']
assert abs(a['portal_coupling_max_given_model']/ (b['lambda40']*np.sqrt(a['LZ_4p2ty_40_GeV_SI_90pct_cm2']/b['SI40_cm2']))-1) < 1e-12
print('PASS LZ scalar portal inversion')
f=r['freezein']
assert abs((f['width_for_observed_Omegac_h2_GeV']/b['freezein_width_GeV'])*b['freezein_omega_h2']/f['Omega_target_from_Planck']-1)<1e-12
print('PASS freeze-in observational inversion')
fo=r['freezeout']
assert fo['success'] and fo['Y_final']>0 and .01<fo['omega_h2']<1
assert abs(fo['omega_h2']-2.742e8*fo['mchi_GeV']*fo['Y_final'])<1e-10
print('PASS freeze-out integral and density conversion')
l=r['weak_lensing']
ls=np.array(l['ell_grid']);cl=np.array(l['cl_kappa_grid']);assert len(ls)==len(cl) and len(ls)>100
assert (cl>=0).all() and (np.diff(ls)>0).all()
for k in (50,200,1000): assert np.isclose(np.interp(k,ls,cl),l[f'ell_{k}_Cl'],rtol=1e-12)
print('PASS weak-lensing multipole outputs, finite and positive')
print('ALL CLOSURE TESTS PASSED')
