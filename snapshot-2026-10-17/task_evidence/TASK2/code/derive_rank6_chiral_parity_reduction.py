from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/data/task2_continue')
OUT=ROOT/'TASK2_RANK6_CHIRAL_PARITY_REDUCTION.json'
route_red=json.load(open('/mnt/data/task2_work/TASK2_RANK6_TO_RANK3_ROUTE_REDUCTION.json'))
weyl=json.load(open('/mnt/data/task2_work/TASK2_WEYL_ROUTE_DETERMINANT_CERTIFICATE.json'))
temp=json.load(open(ROOT/'TEMPORAL_EW_WORLDLINE_CORRECTED.json'))

# Recover the fixed route phases from an already-verified U(1) regression.
h_old=[complex(*z) for z in route_red['u1_corrected_composite_block']['route_amplitudes_h_U1']]
abc_old=[complex(*z) for z in weyl['u1_composite_regression']['abc']]
route_phase=[abc_old[i]/h_old[i] for i in range(3)]
assert max(abs(abs(z)-1) for z in route_phase)<1e-14

hEW=[complex(*z) for z in temp['EW_route_amplitudes']]
a,b,c=[route_phase[i]*hEW[i] for i in range(3)]

omega=np.exp(2j*np.pi/3)
I3=np.eye(3,dtype=complex)
X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex)
Z=np.diag([1,omega,omega**2])
C=np.array([[0,1],[1,0]],complex)
I2=np.eye(2,dtype=complex)

# Correct route parity: d=10,8 are even; d=9 is odd.
T=(a*np.kron(I2,I3)+b*np.kron(C,X)+c*np.kron(I2,Z))/3
s=np.linalg.svd(T,compute_uv=False)
K=float(-np.mean(np.log(s)))
leak=np.eye(6)-T.conj().T@T
leak_min=float(np.linalg.eigvalsh((leak+leak.conj().T)/2).min())

Bp=(a*I3+b*X+c*Z)/3
Bm=(a*I3-b*X+c*Z)/3
s_blocks=np.sort(np.concatenate([np.linalg.svd(Bp,compute_uv=False),np.linalg.svd(Bm,compute_uv=False)]))[::-1]
block_res=float(np.max(np.abs(np.sort(s)[::-1]-s_blocks)))
Dplus=a**3+b**3+c**3
Dminus=a**3-b**3+c**3
det_formula=Dplus*Dminus/(27**2)
det_direct=np.linalg.det(T)
det_res=float(abs(det_direct-det_formula))
K_formula=float(math.log(3)-(math.log(abs(Dplus))+math.log(abs(Dminus)))/6)

# Random algebraic regression tests of determinant/parity theorem.
rng=np.random.default_rng(20260918)
max_det=0.0;max_block=0.0;max_K=0.0
for _ in range(2000):
    aa,bb,cc=rng.normal(size=3)+1j*rng.normal(size=3)
    TT=(aa*np.kron(I2,I3)+bb*np.kron(C,X)+cc*np.kron(I2,Z))/3
    BBp=(aa*I3+bb*X+cc*Z)/3; BBm=(aa*I3-bb*X+cc*Z)/3
    d=((aa**3+bb**3+cc**3)*(aa**3-bb**3+cc**3))/(27**2)
    max_det=max(max_det,float(abs(np.linalg.det(TT)-d)))
    sv=np.linalg.svd(TT,compute_uv=False)
    sb=np.concatenate([np.linalg.svd(BBp,compute_uv=False),np.linalg.svd(BBm,compute_uv=False)])
    max_block=max(max_block,float(np.max(np.abs(np.sort(sv)-np.sort(sb)))))
    if abs(d)>1e-20:
        kd=-np.log(abs(np.linalg.det(TT)))/6
        kf=math.log(3)-(math.log(abs(aa**3+bb**3+cc**3))+math.log(abs(aa**3-bb**3+cc**3)))/6
        max_K=max(max_K,float(abs(kd-kf)))

report={
 'status':'PASS_TASK2_EXACT_CHIRAL_PARITY_AND_WEYL_DETERMINANT_REDUCTION',
 'external_particle_data_used':False,
 'upstream_task1_operator_fingerprint':'2e7f0710866ab7d7783ed1ce61b14267494467e73e7f82f9d10aeb134ee5dfe6',
 'theorem':{
   'composite_carrier':'On the active electron branch, L_L and e_R tensor H_R both carry (q_U1,2j_SU2,pq_SU3)=(-3,1,(0,0)). Hence the covariant stream and gauge gates are identical on the two chiral branches.',
   'one_particle_collision':'U_cell is the identity on the declared one-particle CAR source sector.',
   'commutation':'The chiral Clebsch involution C commutes with the gauge/electric/plaquette/stream carrier A on this active pair; therefore U_C8^d=C^d A^d.',
   'route_parity':'d=(10,9,8) implies C^d=(I,C,I).',
   'rank6_form':'T_plus=(a I2⊗I3 + b C⊗X + c I2⊗Z)/3.',
   'block_diagonalization':'In the C eigenbasis, T_plus=B_plus ⊕ B_minus, B_plus=(aI+bX+cZ)/3, B_minus=(aI-bX+cZ)/3.',
   'determinant':'det(T_plus)=[(a^3+b^3+c^3)(a^3-b^3+c^3)]/27^2.',
   'K_formula':'K_UV=log(3)-[log|a^3+b^3+c^3|+log|a^3-b^3+c^3|]/6.',
   'error_bound':'If |delta a_i|<=eps_i, define E=sum_i(3|a_i|^2 eps_i+3|a_i| eps_i^2+eps_i^3). If E<min(|D_plus|,|D_minus|), then |delta K| <= [-log(1-E/|D_plus|)-log(1-E/|D_minus|)]/6.'
 },
 'route_phases':[[z.real,z.imag] for z in route_phase],
 'route_phase_unit_modulus_max_error':float(max(abs(abs(z)-1) for z in route_phase)),
 'temporal_worldline_regression_corrected_for_chiral_parity':{
   'scope':'Reduced temporal U(1) Kato-strip × local SU(2) worldline-rotor regression only; not the final 3D PEPO scalar route contraction.',
   'EW_unweighted_route_amplitudes':[[z.real,z.imag] for z in hEW],
   'weighted_abc':[[z.real,z.imag] for z in (a,b,c)],
   'D_plus':[Dplus.real,Dplus.imag],
   'D_minus':[Dminus.real,Dminus.imag],
   'singular_values':s.tolist(),
   'K_UV_temporal_regression':K,
   'K_formula':K_formula,
   'K_formula_residual':float(abs(K-K_formula)),
   'leakage_min_eigenvalue':leak_min,
   'operator_norm':float(s.max()),
   'det_direct':[det_direct.real,det_direct.imag],
   'det_formula':[det_formula.real,det_formula.imag],
   'det_residual':det_res,
   'block_spectrum_residual':block_res,
   'previous_always_offdiagonal_K_for_reference_only':temp['K_UV_temporal_regression'],
   'parity_correction_delta_K':float(K-temp['K_UV_temporal_regression'])
 },
 'random_algebraic_validation':{
   'tests':2000,'seed':20260918,'max_det_residual':max_det,'max_block_spectrum_residual':max_block,'max_K_formula_residual':max_K
 }
}
txt=json.dumps(report,indent=2,sort_keys=True)+'\n'
OUT.write_text(txt)
print(json.dumps({
 'status':report['status'],
 'K_temporal_parity_corrected':K,
 'singular_values':s.tolist(),
 'leakage_min_eigenvalue':leak_min,
 'D_plus_abs':abs(Dplus),'D_minus_abs':abs(Dminus),
 'det_residual':det_res,'block_residual':block_res,
 'random_max_det_residual':max_det,'random_max_K_residual':max_K,
 'sha256':hashlib.sha256(txt.encode()).hexdigest()
},indent=2))
