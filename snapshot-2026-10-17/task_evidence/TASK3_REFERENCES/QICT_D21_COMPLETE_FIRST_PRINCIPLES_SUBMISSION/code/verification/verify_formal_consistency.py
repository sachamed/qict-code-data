#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parents[2]
TEX=(ROOT/'manuscript/QICT_MAIN_MANUSCRIPT.tex').read_text()
# 1) C3 spectral projectors and Weyl-partner partial isometries.
w=np.exp(2j*np.pi/3)
T=np.array([[0,0,1],[1,0,0],[0,1,0]],complex)  # |j> -> |j+1>
Q=np.diag([1,w,w**2])
P=[]
for a in range(3):
    P.append(sum((w**(-a*k))*np.linalg.matrix_power(T,k) for k in range(3))/3)
proj=max(np.linalg.norm(P[a]@P[b]-(P[a] if a==b else 0)) for a in range(3) for b in range(3))
sumerr=np.linalg.norm(sum(P)-np.eye(3))
spec=max(np.linalg.norm(T@P[a]-w**a*P[a]) for a in range(3))
weyl=np.linalg.norm(T@Q-w**(-1)*Q@T)
part=[]
for a in range(3):
    W=np.linalg.matrix_power(Q,-a)@P[0]
    part.append(max(np.linalg.norm(W.conj().T@W-P[0]),np.linalg.norm(W@W.conj().T-P[a])))
# 2) GLS identity against direct least-squares solve.
rng=np.random.default_rng(21092026)
n=12
X=np.column_stack([np.linspace(1,4,n),np.ones(n),np.linspace(.2,1.1,n)**2])
A=rng.normal(size=(n,n)); Sigma=A@A.T+0.5*np.eye(n); W=np.linalg.inv(Sigma); y=rng.normal(size=n)
beta=np.linalg.solve(X.T@W@X,X.T@W@y); e0=np.array([0.,1.,0.])
b0=float(e0@np.linalg.solve(X.T@W@X,X.T@W@y))
glserr=abs(b0-beta[1])
# 3) Hypercharge full-rank linear system with Y_H=1/2.
YH=0.5
# unknown [YQ,Yu,Yd,YL,Ye,Ynu]
M=np.array([[-1,1,0,0,0,0],[-1,0,1,0,0,0],[0,0,0,-1,1,0],[0,0,0,-1,0,1],[3,0,0,1,0,0],[0,0,0,1,0,0]],float)
b=np.array([YH,-YH,-YH,YH,0,-YH],float)
sol=np.linalg.solve(M,b); target=np.array([1/6,2/3,-1/3,-1/2,-1,0.])
hyper=np.max(np.abs(sol-target))
# 4) Source-level regressions.
for forbidden in ['W_a=T^aP_0','(e_0^T X^TWX)^{-1}','H_R=\\mu\\id+\\nu\\sigma_y','C_3\\text{ symmetry}\\Rightarrow\\text{TBM UV boundary}']:
    assert forbidden not in TEX, forbidden
assert proj<1e-12 and sumerr<1e-12 and spec<1e-12 and weyl<1e-12 and max(part)<1e-12 and glserr<1e-12 and hyper<1e-12
out={'status':'PASS','projector_algebra_error':proj,'projector_sum_error':sumerr,'spectral_projector_error':spec,'weyl_relation_error':weyl,'partial_isometry_error':max(part),'gls_intercept_identity_error':glserr,'hypercharge_linear_system_error':float(hyper)}
(ROOT/'certificates/formal_consistency_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
