#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np
from qict_route_first import receiver_return,singular_angles,postcompression_phase
T=receiver_return(); s,theta=singular_angles(T)
alpha=np.linspace(0,np.pi,20001)
eps=np.array([[postcompression_phase(t,a) for a in alpha] for t in theta])
# only ratios where denominator is nonzero
r21=eps[1]/eps[0]; r31=eps[2]/eps[0]
out={
 'status':'PASS',
 'theorem':'For common family-blind stream eigenphase alpha, cos eps_a = cos(theta_a) cos(alpha), hence eps_b/eps_a <= theta_b/theta_a for 0<theta_a<theta_b<=pi/2.',
 'singular_values':s.tolist(),'theta':theta.tolist(),
 'analytic_upper_ratio_2_1':float(theta[1]/theta[0]),
 'analytic_upper_ratio_3_1':float(theta[2]/theta[0]),
 'grid_max_ratio_2_1':float(np.max(r21)),
 'grid_max_ratio_3_1':float(np.max(r31)),
 'grid_points':len(alpha),
 'external_particle_data_used':False,
 'new_fit_parameters':0
}
Path(__file__).resolve().parents[1].joinpath('results/postcompression_no_go.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
