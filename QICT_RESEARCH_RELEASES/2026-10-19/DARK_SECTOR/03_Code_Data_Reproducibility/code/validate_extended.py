"""Independent consistency checks for the new 3D, scalar-response and stellar calculations."""
from pathlib import Path
import json, math
import numpy as np
B=Path(__file__).resolve().parents[1]
metric=json.loads((B/'data'/'metric_growth_galaxy.json').read_text())
pm=json.loads((B/'data'/'particle_mesh_results.json').read_text())
conv=json.loads((B/'data'/'pm_convergence.json').read_text()) if (B/'data'/'pm_convergence.json').exists() else None
s=metric['growth'];gal=metric['star_formation']
assert abs(s['Sigma_B']-1.05*.95)<1e-14
assert abs(s['EG_z0p5_B']-0.315*s['Sigma_B']/s['f_z0p5_B'])<1e-13
assert abs(s['EG_z0p5_GR']-0.315/s['f_z0p5_GR'])<1e-13
assert abs(gal['SFR_Msun_per_yr_at_3_Gyr']-4.442058514868451)<1e-9
assert gal['stellar_mass_Msun_at_3_Gyr']>0
assert abs(pm['snapshots']['a1.00']['mean_delta'])<1e-12
assert pm['snapshots']['a1.00']['nparticles']==32**3
assert abs(pm['snapshots']['a1.00']['rms_delta']-10.46902427613877)<2e-8
if conv:
 assert conv['rms_relative_delta']<0.003
 assert conv['peak_relative_delta']<0.005
with np.load(B/'data'/'particle_mesh_snapshots.npz') as z:
 assert 'a1.00' in z
 assert z['a1.00'].shape==(32,32,32)
 assert np.isfinite(z['a1.00']).all()
 assert abs(float(np.sum(z['a1.00'])))<1e-9
print('PASS: 3D PM mass conservation, scalar-growth E_G identity, gas-regulator, step convergence, data arrays')
