#!/usr/bin/env python3
"""Machine checks for the exact finite C21 receiver layer.
This script verifies implementations of analytic identities; it is not a substitute for the proofs.
"""
import json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True); CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
N=21
# Shift and cycle Dirac/Laplacian
S=np.roll(np.eye(N,dtype=complex),1,axis=0)
B=np.eye(N)-S
L=B.conj().T@B
D=np.block([[np.zeros((N,N),complex),B.conj().T],[B,np.zeros((N,N),complex)]])
sv=np.sort(np.linalg.svd(B,compute_uv=False))
expected_D=np.sort(np.r_[-sv,sv])
res_D=float(np.max(np.abs(np.linalg.eigvalsh(D)-expected_D)))
expected_L=np.sort(4*np.sin(np.pi*np.arange(N)/N)**2)
res_L=float(np.max(np.abs(np.sort(np.linalg.eigvalsh(L))-expected_L)))
# CRT permutation j -> (a=j mod 3,b=j mod 7), with lexicographic a,b order.
W=np.zeros((N,N),complex)
for j in range(N):
    a=j%3; b=j%7; idx=7*a+b; W[idx,j]=1
S3=np.roll(np.eye(3,dtype=complex),1,axis=0); S7=np.roll(np.eye(7,dtype=complex),1,axis=0)
crt_shift=float(np.linalg.norm(W@S@W.conj().T-np.kron(S3,S7),'fro'))
R=np.linalg.matrix_power(S,7); T=np.linalg.matrix_power(S,15)
crt_R=float(np.linalg.norm(W@R@W.conj().T-np.kron(S3,np.eye(7)),'fro'))
crt_T=float(np.linalg.norm(W@T@W.conj().T-np.kron(np.eye(3),S7),'fro'))
rt=float(np.linalg.norm(R@T-S,'fro'))
# Weyl and active 126-step QCA update.
omega=np.exp(2j*np.pi/3)
X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex); Z=np.diag([1,omega,omega**2])
weyl=float(np.linalg.norm(Z@X-omega*X@Z,'fro'))
Sigma=np.array([[0,1],[1,0]],complex)
# residue projectors on cell coordinate, F onsite on family
Rs=[]
for a in range(3):
    P=np.zeros((N,N),complex)
    for j in range(N):
        if j%3==a: P[j,j]=1
    Rs.append(P)
F=np.kron(Rs[0],np.kron(np.eye(2),np.eye(3)))+np.kron(Rs[1],np.kron(np.eye(2),X))+np.kron(Rs[2],np.kron(np.eye(2),Z))
U=np.kron(S,np.kron(Sigma,np.eye(3)))@F
unit=float(np.linalg.norm(U.conj().T@U-np.eye(126),'fro'))
rec126=float(np.linalg.norm(np.linalg.matrix_power(U,126)-np.eye(126),'fro'))
# eigenvalues should be all 126th roots once; compare sorted principal phases to uniform lattice.
evals=np.linalg.eigvals(U); phases=np.mod(np.angle(evals),2*np.pi); step=2*np.pi/126
# Compare each numerical eigenphase to its nearest 126th-root phase; avoid sorting ambiguity at 0/2pi.
nearest=np.round(phases/step)*step; nearest=np.mod(nearest,2*np.pi)
diff=np.abs(phases-nearest); phase_res=float(np.max(np.minimum(diff,2*np.pi-diff)))
# Exact reciprocal cyclic-average identity on a deterministic nonuniform positive delay profile.
tau=1.0+0.1*np.cos(2*np.pi*np.arange(N)/N)+0.03*np.sin(4*np.pi*np.arange(N)/N)
route_res={}
for d in (10,9,8):
    vals=[]
    for s0 in range(N):
        idx=[(s0+k)%N for k in range(d)]
        mask=np.ones(N,bool); mask[idx]=False
        vals.append(float(np.sum(tau[mask])-np.sum(tau[idx])))
    route_res[str(d)]=float(abs(np.mean(vals)-(N-2*d)*np.mean(tau)))
out={
 'status':'PASS', 'n':N,
 'cycle_dirac_spectrum_max_residual':res_D,
 'cycle_laplacian_spectrum_max_residual':res_L,
 'crt_shift_fro':crt_shift,'crt_R_fro':crt_R,'crt_T_fro':crt_T,'RT_minus_S_fro':rt,
 'weyl_relation_fro':weyl,'U126_unitarity_fro':unit,'U126_recurrence_fro':rec126,
 'uniform_126_phase_lattice_max_residual':phase_res,
 'cyclic_average_route_residuals':route_res,
 'interpretation':'Numerical implementation checks of class-A finite identities; no particle or cosmology data are used.'
}
limits=[res_D<5e-14,res_L<5e-14,crt_shift<1e-13,crt_R<1e-13,crt_T<1e-13,rt<1e-13,weyl<1e-13,unit<1e-12,rec126<5e-11,phase_res<1e-11,max(route_res.values())<1e-12]
out['status']='PASS' if all(limits) else 'FAIL'
(CERT/'exact_cycle_certificate.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
raise SystemExit(0 if out['status']=='PASS' else 1)
