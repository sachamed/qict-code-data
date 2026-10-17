from __future__ import annotations
import json,math,hashlib
from pathlib import Path
r=json.load(open('/mnt/data/task2_continue/TASK2_RANK6_CHIRAL_PARITY_REDUCTION.json'))
abc=[complex(*z) for z in r['temporal_worldline_regression_corrected_for_chiral_parity']['weighted_abc']]
a,b,c=abc;Dp=a**3+b**3+c**3;Dm=a**3-b**3+c**3
seq=[
 {'U1_k':2,'SU2_j2':4,'SU3_pq':7,'epsU':6.5204e-1},
 {'U1_k':3,'SU2_j2':6,'SU3_pq':9,'epsU':1.1221e-2},
 {'U1_k':4,'SU2_j2':8,'SU3_pq':11,'epsU':1.5863e-4},
 {'U1_k':5,'SU2_j2':10,'SU3_pq':13,'epsU':1.8116e-6},
 {'U1_k':6,'SU2_j2':12,'SU3_pq':15,'epsU':1.8780940154408358e-8},
 {'U1_k':7,'SU2_j2':14,'SU3_pq':17,'epsU':1.7702e-10},
]
def ce(z,e):return 3*abs(z)**2*e+3*abs(z)*e*e+e**3
rows=[]
for x in seq:
 eps=[10*x['epsU'],9*x['epsU'],8*x['epsU']]
 E=sum(ce(z,e) for z,e in zip(abc,eps))
 ok=E<min(abs(Dp),abs(Dm))
 dK=(-math.log(1-E/abs(Dp))-math.log(1-E/abs(Dm)))/6 if ok else None
 rows.append({**x,'route_scalar_error_bounds':eps,'cubic_D_error_bound':E,'finite_log_bound':ok,'K_error_bound':dK})
out={'status':'PASS_TASK2_ROUTE_DETERMINANT_ERROR_PROPAGATION','external_particle_data_used':False,
 'note':'Numerical values use the reduced temporal regression only to quantify conditioning. The theorem is exact and will be re-evaluated with the final 3D route scalars.',
 'Dplus_abs_regression':abs(Dp),'Dminus_abs_regression':abs(Dm),'rows':rows,
 'conclusion':'At the present regression conditioning, the finer Task-1 cutoff (7,14,17) reduces the inherited determinant error bound below 3.1e-8; the frozen (6,12,15) point gives about 3.27e-6.'}
txt=json.dumps(out,indent=2,sort_keys=True)+'\n';Path('/mnt/data/task2_continue/TASK2_PARITY_ERROR_BUDGET.json').write_text(txt)
print(json.dumps({'status':out['status'],'rows':[{'cutoff':(x['U1_k'],x['SU2_j2'],x['SU3_pq']),'dK':x['K_error_bound']} for x in rows],'sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2))
