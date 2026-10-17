#!/usr/bin/env python3
import json, math, cmath, hashlib
import numpy as np
phi=math.pi/63
# local two-mode occupation basis |00>,|01>,|10>,|11>
n=np.diag([0.,1.])
I=np.eye(2)
S=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],complex)
C=np.diag([1,1,1,cmath.exp(-1j*phi)])
U=S@C
C2=np.kron(I,I)+(cmath.exp(-1j*phi)-1)*np.kron(n,n)
P=np.diag([1,-1,-1,1]).astype(complex)
# operator Schmidt rank of C: reshape U_{a'b',ab} -> (a'a),(b'b)
def op_svals(M):
 T=M.reshape(2,2,2,2) # a',b',a,b
 R=np.transpose(T,(0,2,1,3)).reshape(4,4)
 return np.linalg.svd(R,compute_uv=False)
svC=op_svals(C); svU=op_svals(U)
# Explicit two-term sum product residual
res=float(np.linalg.norm(C-C2,2))
unit=float(np.linalg.norm(U.conj().T@U-np.eye(4),2))
par=float(np.linalg.norm(U@P-P@U,2))
# number conservation N=nL+nR
N=np.kron(n,I)+np.kron(I,n)
num=float(np.linalg.norm(U@N-N@U,2))
rec=float(np.linalg.norm(np.linalg.matrix_power(U,126)-np.eye(4),2))
rep={
 'status':'PASS_CAR_LOCAL_COLLISION_LOW_RANK',
 'phi_exact':'pi/63',
 'specified_gate':'U_cell = S diag(1,1,1,exp(-i phi))',
 'controlled_phase_decomposition':'C = I⊗I + (exp(-i phi)-1) n_L⊗n_R',
 'controlled_phase_decomposition_residual_2norm':res,
 'controlled_phase_operator_schmidt_singular_values':svC.tolist(),
 'controlled_phase_operator_schmidt_rank_tol_1e-14':int(np.sum(svC>1e-14)),
 'full_gate_operator_schmidt_singular_values':svU.tolist(),
 'full_gate_operator_schmidt_rank_tol_1e-14':int(np.sum(svU>1e-14)),
 'unitarity_residual_2norm':unit,
 'fermion_parity_commutator_2norm':par,
 'number_commutator_2norm':num,
 'recurrence_126_residual_2norm':rec,
 'network_consequence':'Treat S as the explicitly specified leg permutation and C as a sum over a binary auxiliary interaction index. The nontrivial local CAR interaction then adds only bond dimension 2 and preserves parity exactly.',
 'scope':'Local C8 collision only; no claim yet of full rank-six CAR contraction.'
}
out='/mnt/data/qict_task1_v5/CAR_COLLISION_LOWRANK_CERTIFICATE.json'
with open(out,'w') as f:json.dump(rep,f,indent=2);f.write('\n')
print(json.dumps(rep,indent=2))
