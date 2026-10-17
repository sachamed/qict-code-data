#!/usr/bin/env python3
"""Independent verification of the transpose-symmetric Majorana return and bare Nambu continuation.

The script reconstructs the neutral Dirac seed, Majorana return, Takagi singular
spectrum, and the active/sterile continuation gap.  No PMNS matrix is introduced.
"""
from __future__ import annotations
import csv, json
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True); CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
omega=np.exp(2j*np.pi/3)
X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex); Z=np.diag([1,omega,omega**2]); I=np.eye(3,dtype=complex); W=[I,X,Z]
dvals=[10,9,8]; eps={10:Fraction(1,210),9:Fraction(1,63),8:Fraction(5,168)}; phi=Fraction(1,63)

# Neutral Dirac C6 seed.
T=np.zeros((3,3),complex)
for a,d in enumerate(dvals):
    ph=phi*(d*18+(21-d)*18)+eps[d]
    T += np.exp(-1j*np.pi*float(ph))*W[a]/3
L,s,Vh=np.linalg.svd(T); o=np.argsort(s); L=L[:,o]; s=s[o]; R=Vh.conj().T[:,o]
th=np.arcsin(s); Dnu=L@np.diag(th)@R.conj().T

def takagi(M,tol=1e-12):
    H=M.conj().T@M
    vals,U=np.linalg.eigh(H); idx=np.argsort(vals); vals=vals[idx]; U=U[:,idx]
    ss=np.sqrt(np.clip(vals,0,None)); B=U.T@M@U; U2=U.copy()
    for i in range(len(ss)):
        if abs(B[i,i])>tol:
            U2[:,i]*=np.exp(-0.5j*np.angle(B[i,i]))
    return U2,ss,U2.T@M@U2

A=X@(Z@Z); AT=A.T
r=[np.exp(-1j*np.pi*float(eps[d]))*(omega**a) for a,d in enumerate(dvals)]
Traw=(r[0]*I+r[1]*A+r[2]*AT)/3
TM=(Traw+Traw.T)/2
alpha=r[0]/3; beta=(r[1]+r[2])/6
proj_res=float(np.linalg.norm(TM-(alpha*I+beta*(A+AT)),'fro'))
sym_TM=float(np.linalg.norm(TM-TM.T,'fro'))
UM,sM,BM=takagi(TM); thM=np.arcsin(np.clip(sM,0,1))
MR=UM.conj()@np.diag(thM)@UM.conj().T
Z3=np.zeros((3,3),complex)
MNG=np.block([[Z3,Dnu],[Dnu.T,MR]])
UN,sN,BN=takagi(MNG); active=sN[:3]; sterile=sN[3:]

# Dense-grid + Lipschitz lower-bound certificate for continuity of the three lower branches.
gridN=10001; h=1/(gridN-1); min_sample=1e9; at=0.0
for k in range(gridN):
    lam=k*h
    M=np.block([[Z3,lam*Dnu],[lam*Dnu.T,MR]])
    ss=np.sort(np.linalg.svd(M,compute_uv=False)); g=float(ss[3]-ss[2])
    if g<min_sample: min_sample=g; at=lam
Lip=float(np.linalg.svd(Dnu,compute_uv=False)[0]); lower=min_sample-Lip*h

expected={}
for row in csv.DictReader(open(DATA/'neutral_spectral_invariants.csv')):
    expected.setdefault(row['class'],[]).append(float(row['spectral_value_over_Etau']))
act_res=max(abs(a-b) for a,b in zip(active,expected['B5_active_continuation_Nambu']))
ster_res=max(abs(a-b) for a,b in zip(sterile,expected['B5_sterile_continuation_Nambu']))
maj_res=max(abs(a-b) for a,b in zip(thM,expected['B5_Majorana_generator']))

# Write the route coefficients in publication form.
with open(DATA/'majorana_route_coefficients.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['route','d','residual_phase_over_pi','cp_orbit_real','cp_orbit_imag','coefficient_real','coefficient_imag'])
    for a,d in enumerate(dvals):
        w.writerow([a,d,float(eps[d]),float((omega**a).real),float((omega**a).imag),float(r[a].real),float(r[a].imag)])

out={
 'status':'PASS',
 'closure_class':'B5',
 'alpha':{'real':float(alpha.real),'imag':float(alpha.imag)},
 'beta':{'real':float(beta.real),'imag':float(beta.imag)},
 'majorana_projection_formula_fro':proj_res,
 'T_M_symmetry_fro':sym_TM,
 'majorana_return_singular_values':[float(x) for x in sM],
 'majorana_generator_spectral_values_over_Etau':[float(x) for x in thM],
 'active_continuation_spectral_values_over_Etau':[float(x) for x in active],
 'sterile_continuation_spectral_values_over_Etau':[float(x) for x in sterile],
 'active_table_max_residual':act_res,
 'sterile_table_max_residual':ster_res,
 'majorana_table_max_residual':maj_res,
 'homotopy_gap':{'grid_points':gridN,'min_sample_gap':min_sample,'at_lambda':at,
                 'Dnu_operator_norm':Lip,'certified_lower_bound_by_Lipschitz':lower},
 'interpretation':'The B5 construction fixes the bare neutral singular spectrum and its separated three-branch continuation. It does not determine the physical PMNS matrix.'
}
out['status']='PASS' if proj_res<1e-13 and sym_TM<1e-13 and act_res<2e-13 and ster_res<2e-13 and maj_res<2e-13 and lower>0 else 'FAIL'
(CERT/'neutral_nambu_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if out['status']=='PASS' else 1)
