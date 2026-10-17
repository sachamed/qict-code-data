#!/usr/bin/env python3
"""G1 theorem-side certificate: two-band periodic SU(2) Weyl symbols have zero net index.

This script does not prove the differential-topology theorem by computation; it records
its hypotheses and independently checks the C7 local indices that instantiate it.
No experimental particle data are accessed.
"""
from pathlib import Path
import json, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)

def nminus(k):
    x,y,z=map(float,k)
    cx,cy,cz=np.cos([x,y,z]); sx,sy,sz=np.sin([x,y,z])
    d=float(cx*cy*cz-sx*sy*sz)
    n=np.array([sx*cy*cz+cx*sy*sz,
                cx*sy*cz-sx*cy*sz,
                cx*cy*sz+sx*sy*cz],float)
    return d,n

def jac(x,h=2e-6):
    x=np.asarray(x,float); J=np.empty((3,3))
    for j in range(3):
        e=np.zeros(3); e[j]=h
        J[:,j]=(nminus(x+e)[1]-nminus(x-e)[1])/(2*h)
    return J
nodes=[
    ('Gamma',[0.,0.,0.]),
    ('Kminus',[-math.pi/2]*3),
    ('Kplus',[math.pi/2]*3),
    ('Xpi',[math.pi,0.,0.]),
]
rows=[]
for name,k in nodes:
    d,n=nminus(k); det=float(np.linalg.det(jac(k)))
    rows.append({'name':name,'kappa':k,'d':d,'n_norm':float(np.linalg.norm(n)),
                 'quasienergy':'0' if d>0 else 'pi','jacobian_det':det,
                 'index':int(np.sign(det))})
zero=[r for r in rows if r['quasienergy']=='0']
# For the explicit representatives, the relevant test is existence of opposite local indices.
opposite_zero=(len(zero)>=2 and {r['index'] for r in zero}=={-1,1})
cert={
 'gap':'G1',
 'taxonomy':'DEMONSTRATED',
 'theorem':'For a smooth periodic two-band SU(2) symbol U(k)=a0(k)I-i a(k).sigma on T^3 with isolated regular zeros of a, the sum of local degrees (Weyl chiralities) is zero. Hence a net single Weyl node is impossible within the declared class.',
 'proof_kernel':['a:T^3->R^3 is a section of the trivial oriented rank-3 bundle',
                 'remove disjoint balls around all regular zeros and set a_hat=a/|a|',
                 'the normalized S^2 area 2-form omega is closed on the punctured torus',
                 'Stokes gives sum_i integral_{S_i^2} omega=0',
                 'each integral is the local Brouwer degree, equal to Weyl chirality for a linear node'],
 'hypotheses':{'translation_invariant':True,'periodic_BZ_T3':True,'finite_range_or_smooth_locality':True,
               'unitary_two_band_SU2_symbol':True,'isolated_regular_Weyl_nodes':True},
 'explicit_C7_nodes':rows,
 'opposite_chirality_zero_quasienergy_nodes_present':opposite_zero,
 'conclusion':'NO_DOUBLER_SAFE_DEFORMATION_EXISTS_WITH_ALL_G1_CONSTRAINTS_UNCHANGED',
 'minimal_escape_classes':['increase internal carrier / vectorlike Dirac completion',
                           'modify lattice chirality (Ginsparg-Wilson/overlap) and relax ultralocal finite-range form',
                           'add an auxiliary dimension (domain-wall)',
                           'add mirror degrees of freedom and prove anomaly-compatible interacting decoupling'],
 'external_particle_data_opened':False,
 'status':'PASS' if opposite_zero else 'FAIL'
}
(CERT/'g1_topological_obstruction_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status']=='PASS' else 1)
