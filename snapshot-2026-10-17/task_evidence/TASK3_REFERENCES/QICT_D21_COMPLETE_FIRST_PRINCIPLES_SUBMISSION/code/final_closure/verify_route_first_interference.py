#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np
from qict_route_first import electron_route_coefficients,receiver_return,analytic_zero_holonomy
c=electron_route_coefficients(); h=analytic_zero_holonomy(c)
T0=receiver_return(c,np.ones(3,complex)); Tz=receiver_return(c,h)
# determinant identity det[(aI+bX+cZ)/3]=(a^3+b^3+c^3)/27
abc=c*h
rhs=(abc[0]**3+abc[1]**3+abc[2]**3)/27
sv0=np.sort(np.linalg.svd(T0,compute_uv=False))
svz=np.sort(np.linalg.svd(Tz,compute_uv=False))
# projector variation along exact zero-holonomy path
ph=np.unwrap(np.angle(h))
vals=[]
prev=None
for t in np.linspace(0,1,401):
    ht=np.exp(1j*t*ph)
    T=receiver_return(c,ht)
    L,s,Rh=np.linalg.svd(T)
    order=np.argsort(s); s=s[order]; L=L[:,order]
    P=np.outer(L[:,0],L[:,0].conj())
    if prev is not None: vals.append(np.linalg.norm(P-prev,2))
    prev=P
out={
 'status':'PASS',
 'identity_residual':float(abs(np.linalg.det(Tz)-rhs)),
 'determinant_at_analytic_zero_abs':float(abs(np.linalg.det(Tz))),
 'holonomy_modulus_defect':float(np.max(abs(abs(h)-1))),
 'flat_singular_values':sv0.tolist(),
 'zero_holonomy_singular_values':svz.tolist(),
 'smallest_singular_value':float(svz[0]),
 'largest_adjacent_projector_change_along_path':float(max(vals)),
 'analytic_holonomy_phases':np.angle(h).tolist(),
 'external_particle_data_used':False,
 'new_fit_parameters':0,
 'interpretation':'Capacity theorem only: route-first holonomy can create an exact dark singular channel. This is not a fitted physical vacuum or a lepton-mass prediction.'
}
Path(__file__).resolve().parents[1].joinpath('results/route_first_interference.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
