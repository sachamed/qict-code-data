#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction as F
import cmath,json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
main=(ROOT/'manuscript/QICT_MAIN_MANUSCRIPT.tex').read_text(errors='ignore')
x=(ROOT/'supplements/QICT_Supplement_X_Assumptions_Global_Gauge_Consistency_and_Convergence.tex').read_text(errors='ignore')
# One-generation LH anomaly ledger
Y={'Q':F(1,6),'uc':F(-2,3),'dc':F(1,3),'L':F(-1,2),'ec':F(1),'nc':F(0)}
A33=2-1-1
A331=2*Y['Q']+Y['uc']+Y['dc']
A221=3*Y['Q']+Y['L']
Agr=6*Y['Q']+3*Y['uc']+3*Y['dc']+2*Y['L']+Y['ec']+Y['nc']
A111=6*Y['Q']**3+3*Y['uc']**3+3*Y['dc']**3+2*Y['L']**3+Y['ec']**3+Y['nc']**3
witten_doublets=4
# Z6 quotient on physical multiplets using q=6Y and generator (omega3,-1,e^{i pi/3})
reps={
 'Q':(1,1,1), 'uR':(1,0,4), 'dR':(1,0,-2), 'L':(0,1,-3), 'eR':(0,0,-6), 'nuR':(0,0,0), 'H':(0,1,3)
}
quot={}
for name,(triality,doublet,q6) in reps.items():
    z=cmath.exp(2j*cmath.pi*triality/3)*((-1)**doublet)*cmath.exp(1j*cmath.pi*q6/3)
    quot[name]=abs(z-1)
# Weyl basis rank and HS orthogonality
omega=np.exp(2j*np.pi/3); X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex); Z=np.diag([1,omega,omega**2])
basis=[]
for a in range(3):
    for b in range(3): basis.append(np.linalg.matrix_power(X,a)@np.linalg.matrix_power(Z,b))
M=np.column_stack([B.reshape(-1) for B in basis]); rank=int(np.linalg.matrix_rank(M,tol=1e-11))
G=np.array([[np.trace(A.conj().T@B) for B in basis] for A in basis]); hs=float(np.max(np.abs(G-3*np.eye(9))))
# Assumption and scope guardrails
assumptions=all(re.search(rf'\bA{i}\b',x) for i in range(1,9)) and all(tag in x for tag in ['LP1','LP2','GQ','RC','MC'])
scope_tests={
 'class-relative uniqueness': (
     'uniqueness statements are restricted to the explicitly declared microscopic classes' in main.lower()
     or 'all uniqueness and minimality statements are relative to this explicitly defined microscopic class' in main.lower()
 ),
 'global non-Abelian chiral measure': 'global non-abelian chiral measure' in main.lower() or 'global non-abelian chiral measure' in x.lower(),
 'mirror decoupling': 'mirror decoupling' in main.lower() or 'mirror decoupling' in x.lower(),
 'rest energies': 'rest energies' in main.lower() or 'rest energies' in x.lower(),
 'CKM/PMNS': 'ckm/pmns' in main.lower() or 'ckm/pmns' in x.lower(),
 'universal-response null hypothesis': 'universal-response null hypothesis' in main.lower() or 'universal-response null hypothesis' in x.lower(),
 'conditional retrospective': 'conditional retrospective' in main.lower() or 'conditional retrospective' in x.lower(),
}
phrase_ok=scope_tests
checks={
 'local_anomalies_exact_zero': all(v==0 for v in [A33,A331,A221,Agr,A111]),
 'witten_even': witten_doublets%2==0,
 'z6_quotient_compatible': max(quot.values())<1e-12,
 'weyl_basis_rank_9': rank==9 and hs<1e-12,
 'explicit_A1_A8_LP1_LP2_plus_GQ_RC_MC': assumptions,
 'scope_guardrails_present': all(phrase_ok.values()),
 'physical_observable_scope_explicit': ('rest energies require regulator-independent interacting spectral edges' in main.lower() and 'species dependent' in x.lower()),
 'receiver_current_observable_scope_explicit': ('receiver-current protocol' in x.lower() and 'distinct from standard ckm/pmns parameters' in x.lower()),
}
status='PASS' if all(checks.values()) else 'FAIL'
out={'status':status,'checks':checks,'anomalies':{'SU3^3':str(A33),'SU3^2U1':str(A331),'SU2^2U1':str(A221),'grav2U1':str(Agr),'U1^3':str(A111)},'witten_doublets':witten_doublets,'z6_residuals':quot,'weyl_rank':rank,'weyl_HS_max_residual':hs,'scope_phrases':phrase_ok}
(ROOT/'certificates/logical_scope_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if status=='PASS' else 1)
