#!/usr/bin/env python3
"""Independent verification of the receiver-route contractions and C6 Julia spectral bridge.

The calculation uses only the specified route lengths, residual phases, Weyl support,
and oriented representation costs.  It verifies the principal-angle spectra and the
bare charged left-basis overlap against the publication data tables.  No experimental
particle data are read.
"""
from __future__ import annotations
import csv, json, math
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
omega=np.exp(2j*np.pi/3)
X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex)
Z=np.diag([1,omega,omega**2]); I=np.eye(3,dtype=complex); W=[I,X,Z]
dvals=[10,9,8]
eps={10:Fraction(1,210),9:Fraction(1,63),8:Fraction(5,168)}
phi=Fraction(1,63)
costs={'u':(10,18),'d':(10,24),'e':(18,24),'nu':(18,18)}

sector={}; Lbasis={}; max_julia_unit=0.0; max_compress=0.0
for sec,(cL,cH) in costs.items():
    T=np.zeros((3,3),complex)
    for a,d in enumerate(dvals):
        ph=phi*(d*cL+(21-d)*cH)+eps[d]
        T += np.exp(-1j*np.pi*float(ph))*W[a]/3
    L,s,Vh=np.linalg.svd(T)
    order=np.argsort(s); L=L[:,order]; s=s[order]; R=Vh.conj().T[:,order]
    theta=np.arcsin(np.clip(s,0,1))
    C=np.diag(np.sqrt(1-s*s)); S=np.diag(s)
    U0=np.block([[C,S],[-S,C]])
    B=np.block([[L,np.zeros((3,3),complex)],[np.zeros((3,3),complex),R]])
    UJ=B@U0@B.conj().T
    max_julia_unit=max(max_julia_unit,float(np.linalg.norm(UJ.conj().T@UJ-np.eye(6),'fro')))
    max_compress=max(max_compress,float(np.linalg.norm(UJ[:3,3:]-T,'fro')))
    sector[sec]={'T_singular_values':[float(x) for x in s],
                 'finite_spectral_values_over_Etau':[float(x) for x in theta]}
    Lbasis[sec]=L

V=Lbasis['u'].conj().T@Lbasis['d']
def invariants(V):
    s13=float(abs(V[0,2])); c13=math.sqrt(max(0,1-s13*s13))
    s12=float(abs(V[0,1])/c13); s23=float(abs(V[1,2])/c13)
    c12=math.sqrt(max(0,1-s12*s12)); c23=math.sqrt(max(0,1-s23*s23))
    J=float(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0])))
    den=c12*c23*c13*c13*s12*s23*s13
    sd=J/den
    num=abs(V[1,0])**2-s12*s12*c23*c23-c12*c12*s23*s23*s13*s13
    den2=2*s12*c12*c23*s23*s13
    cd=float(num/den2)
    delta=math.atan2(sd,cd); delta=delta if delta>=0 else delta+2*math.pi
    return {'theta12_rad':math.asin(s12),'theta13_rad':math.asin(s13),
            'theta23_rad':math.asin(s23),'delta_rad':delta,'J':J,
            's12':s12,'s13':s13,'s23':s23}
inv=invariants(V)

# Compare to independently stored publication tables.
rows=list(csv.DictReader(open(DATA/'finite_spectral_invariants.csv')))
expected={}
for r in rows:
    sec=r['sector']
    if sec in costs and r['class']=='C6_charged_finite_Dirac':
        expected.setdefault(sec,[]).append(float(r['spectral_value_over_Etau']))
# Charged sectors are compared with the finite spectral table; the neutral Dirac seed
# is compared with the C7 rest-angle table because the final finite-spectrum table
# reports the Nambu continuation for the physical neutral block.
max_spec=max(max(abs(a-b) for a,b in zip(sector[s]['finite_spectral_values_over_Etau'],expected[s])) for s in ('u','d','e'))
c7nu=[float(r['principal_angle_rad']) for r in csv.DictReader(open(DATA/'c7_rest_quasienergy_angles.csv')) if r['sector']=='nu']
max_spec=max(max_spec,max(abs(a-b) for a,b in zip(sector['nu']['finite_spectral_values_over_Etau'],c7nu)))

mod=np.zeros((3,3),float)
for r in csv.DictReader(open(DATA/'bare_charged_left_overlap_matrix.csv')):
    mod[int(r['row'])-1,int(r['col'])-1]=float(r['abs_Vch0'])
overlap_res=float(np.linalg.norm(np.abs(V)-mod,'fro'))
inv_expected={r['invariant']:float(r['value']) for r in csv.DictReader(open(DATA/'bare_charged_invariants.csv'))}
inv_res=max(abs(inv[k]-inv_expected[k]) for k in inv)

out={
 'status':'PASS',
 'closure_class':'receiver-route+C6',
 'sector_results':sector,
 'max_finite_spectral_table_residual':max_spec,
 'julia_unitarity_max_fro':max_julia_unit,
 'julia_compression_max_fro':max_compress,
 'bare_charged_left_overlap_abs':np.abs(V).tolist(),
 'bare_overlap_modulus_table_fro':overlap_res,
 'bare_overlap_invariant_max_residual':inv_res,
 'interpretation':'Finite-generator principal-angle invariants and bare basis overlap only; no renormalized fermion mass or physical CKM value is inferred.'
}
out['status']='PASS' if max_spec<2e-14 and max_julia_unit<1e-13 and max_compress<1e-13 and overlap_res<2e-14 and inv_res<2e-14 else 'FAIL'
(CERT/'finite_spectral_bridge_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if out['status']=='PASS' else 1)
