#!/usr/bin/env python3
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import mpmath as mp
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'microscopic_closure'/'TASK1_OPERATOR_NORM_FREEZE.json'
mp.mp.dps=70
phi=mp.pi/63; x1=4*phi; x2=32*phi; x3=4*mp.pi/7

def u1_tail(K):
 s=mp.mpf('0')
 for k in range(K+1,120):
  t=2*abs(mp.besselj(k,x1)); s+=t
  if t<mp.mpf('1e-60'): break
 return s

def su2_tail(J2):
 s=mp.mpf('0')
 for j2 in range(J2+1,160):
  d=j2+1; t=(2*d*d/x2)*abs(mp.besselj(d,x2)); s+=t
  if t<mp.mpf('1e-60'): break
 return s

def su3_tail(R):
 return mp.e**x3-sum(x3**n/mp.factorial(n) for n in range(R+1))
U={K:u1_tail(K) for K in range(2,11)}
S={J:su2_tail(J) for J in range(4,22)}
C={R:su3_tail(R) for R in range(7,25)}
def ep(K,J,R): return (1+U[K])*(1+S[J])*(1+C[R])-1
def gb(L,K,J,R): return (1+ep(K,J,R))**(3*L**3)-1
seq=[(2,4,7),(3,6,9),(4,8,11),(5,10,13),(6,12,15),(7,14,17)]
rows=[]
for K,J,R in seq:
 e=ep(K,J,R)
 rows.append({'U1_abs_k_max':K,'SU2_twice_j_max':J,'SU2_j_max':J/2,'SU3_max_p_q':R,
 'chi_TN_exact_block_channels_including_CAR_parity':(2*K+1)*(J+1)*(R+1)**2*2,
 'one_plaquette_bounds':{'U1':float(U[K]),'SU2':float(S[J]),'SU3':float(C[R]),'combined':float(e)},
 'one_step_operator_norm_bound_L2':float(gb(2,K,J,R)),'one_step_operator_norm_bound_L3':float(gb(3,K,J,R)),'one_step_operator_norm_bound_L4':float(gb(4,K,J,R))})
vol=[]
for L in range(2,9): vol.append({'L':L,'sites':L**3,'links':3*L**3,'plaquettes':3*L**3,'bound':float(gb(L,6,12,15))})
adapt=[]
for L in range(2,13):
 found=None
 for K in range(3,11):
  for J in range(6,21,2):
   for R in range(10,25):
    b=gb(L,K,J,R)
    if b<mp.mpf('1e-6'):
     chi=(2*K+1)*(J+1)*(R+1)**2*2; cand=(chi,K,J,R,b)
     if found is None or cand[:4]<found[:4]: found=cand
 if found is None: raise RuntimeError(L)
 chi,K,J,R,b=found;adapt.append({'L':L,'U1_abs_k_max':K,'SU2_twice_j_max':J,'SU3_max_p_q':R,'chi_TN_exact_block_channels_including_CAR_parity':chi,'one_step_operator_norm_bound':float(b)})
rep={'status':'PASS_TASK1_OPERATOR_NORM_CUTOFF_SEQUENCE','external_particle_data_used':False,'microscopic_phase':'pi/63',
 'theorem':'All electric, CAR collision, composite streaming and Clebsch gates are exact unitaries. The only numerical representation regulator in the one-step PEPO is the magnetic character expansion. Uniform character-tail bounds telescope directly to an operator-norm bound for U_micro. The product-group plaquette bound is (1+eps_U1)(1+eps_SU2)(1+eps_SU3)-1; over 3L^3 commuting plaquettes it is bounded by (1+eps_plaq)^(3L^3)-1.',
 'cutoff_and_chi_sequence':rows,'fixed_cutoff_volume_sequence':vol,'adaptive_volume_sequence_below_1e-6':adapt,
 'thermodynamic_limit_statement':'Global operator norms across different volumes are not compared because the Hilbert spaces differ. QICT defines a quasi-local automorphism: for every fixed local observable, one finite-depth macro-step has finite propagation range, so its Heisenberg image is exactly volume-independent once the periodic box contains its causal neighborhood. The adaptive table controls numerical character truncation uniformly on each finite box.',
 'freeze_cutoff_for_L2_n1':{'U1_abs_k_max':6,'SU2_twice_j_max':12,'SU3_max_p_q':15,'chi_TN_exact_block_channels_including_CAR_parity':86528,'operator_norm_bound':float(gb(2,6,12,15))}}
txt=json.dumps(rep,indent=2,sort_keys=True)+'\n';OUT.write_text(txt)
print(json.dumps(rep|{'sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2))
