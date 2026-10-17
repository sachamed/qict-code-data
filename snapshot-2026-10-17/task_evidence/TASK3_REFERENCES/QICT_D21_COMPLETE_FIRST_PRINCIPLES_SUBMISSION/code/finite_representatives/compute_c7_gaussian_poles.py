#!/usr/bin/env python3
import json, math, csv
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
cert=json.loads((ROOT/'certificates/c7_dirac_lift_certificate.json').read_text())
sectors=cert['matrix_julia_dirac_lift']
rows=[]
for f in ['u','d','e','nu']:
    s=np.array(sectors[f]['transfer_singular_values'],float)
    th=np.arcsin(s)
    c=np.sqrt(1-s*s)
    for a,(si,ti,ci) in enumerate(zip(s,th,c),1):
        rows.append(dict(sector=f,branch=a,s=si,theta=ti,cos_theta=ci,
                         gaussian_rest_energy_over_Etau=ti,
                         gaussian_ZL=1.0,gaussian_ZR=1.0,
                         interacting_physical_mass_status='NOT_EVALUATED'))

# Ratios within each charged sector (dimensionless Gaussian pole ratios)
ratios={}
for f in ['u','d','e','nu']:
    th=np.array(sectors[f]['principal_rest_angles'],float)
    ratios[f]=(th/th[0]).tolist()

# Exact dispersion verification in the singular basis:
# U = [[c A, i s I],[i s I, c A^dag]], det roots obey cos eps = c d(k).

def A_bcc(k, branch='+'):
    kx,ky,kz=k
    cx,cy,cz=[math.cos(v/math.sqrt(3)) for v in k]
    sx,sy,sz=[math.sin(v/math.sqrt(3)) for v in k]
    if branch=='+':
        nx=sx*cy*cz-cx*sy*sz
        ny=-cx*sy*cz-sx*cy*sz
        nz=cx*cy*sz-sx*sy*cz
        d=cx*cy*cz+sx*sy*sz
    else:
        nx=sx*cy*cz+cx*sy*sz
        ny=+cx*sy*cz-sx*cy*sz
        nz=cx*cy*sz+sx*sy*cz
        d=cx*cy*cz-sx*sy*sz
    sig=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.array([[1,0],[0,-1]],complex)]
    A=d*np.eye(2,dtype=complex)-1j*(nx*sig[0]+ny*sig[1]+nz*sig[2])
    return A,d

max_res=0.0
samples=[(0,0,0),(0.2,-0.1,0.15),(0.7,0.3,-0.4),(-0.8,0.5,0.2)]
for branch in ['+','-']:
    for k in samples:
        A,d=A_bcc(k,branch)
        for f in ['u','d','e','nu']:
            for s in sectors[f]['transfer_singular_values']:
                c=math.sqrt(1-s*s)
                U=np.block([[c*A,1j*s*np.eye(2)],[1j*s*np.eye(2),c*A.conj().T]])
                ev=np.linalg.eigvals(U)
                # For every eigenvalue, (z+z^-1)/2 should equal c*d.
                res=max(abs((z+1/z)/2-c*d) for z in ev)
                max_res=max(max_res,float(res))

out={
  'theory':'QICT C7 Gaussian sector',
  'approximation':'Gaussian/bare C7 matrix-Julia sector only',
  'exact_gaussian_inverse_kernel':'Gamma_f0(z,k)=z I_12-U_f0^(7)(k)',
  'exact_gaussian_dispersion':'cos(epsilon_fa(k))=cos(theta_fa)*d_k, theta_fa=arcsin(s_fa)',
  'max_dispersion_identity_residual':max_res,
  'gaussian_wavefunction_coefficients':'Z_L=Z_R=I in the canonically normalized IR bare expansion; interacting residue matrices are not evaluated',
  'interacting_self_energy_status':'NOT EVALUATED: the CTP functional is defined but contains no completed exact/perturbative solution of its interacting many-body/gauge correlators',
  'neutral_status':'The bare Nambu singular spectrum is specified, but the model does not give an explicit matrix-Julia C7 unitary for the full 6x6 Dirac+Majorana Nambu kernel; therefore no explicit neutral C7 Gaussian pole operator is inferred here.',
  'absolute_scale_status':'NOT IDENTIFIABLE DATA-BLIND. the one-clock theorem requires an independently measured nonzero receiver delay tau_*. A charged-lepton mass calibration would be non-blind.',
  'physical_mass_status':'NOT NUMERICALLY DETERMINED',
  'gaussian_dimensionless_ratios':ratios,
  'rows':rows,
}
(ROOT/'certificates/c7_gaussian_pole_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
with open(ROOT/'data/c7_gaussian_poles.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(json.dumps(out,indent=2))
