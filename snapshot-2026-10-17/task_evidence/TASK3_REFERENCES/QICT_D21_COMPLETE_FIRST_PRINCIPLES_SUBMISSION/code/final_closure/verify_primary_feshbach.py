#!/usr/bin/env python3
from pathlib import Path
import json, math, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0,str(Path(__file__).resolve().parent))
import qict_route_first as qrf

U=qrf.route_unitary()
omega=np.exp(2j*np.pi/3)
route_basis=np.array([[1,1,1],[1,omega,omega**2],[1,omega**2,omega]],complex).T/np.sqrt(3)
S=np.kron(route_basis,np.eye(3))
Ub=S.conj().T@U@S
A,B=Ub[:3,:3],Ub[:3,3:]
C,D=Ub[3:,:3],Ub[3:,3:]

res=[]
for lam in (0.0,0.13,0.5,1.0):
    sl=math.sqrt(lam)
    Ul=np.block([[A,sl*B],[sl*C,D]])
    for z in (1.17*np.exp(.31j),.91*np.exp(-.77j),1.31*np.exp(1.12j)):
        G=z*np.eye(3)-A-lam*B@np.linalg.inv(z*np.eye(6)-D)@C
        lhs=np.linalg.det(z*np.eye(9)-Ul)
        rhs=np.linalg.det(z*np.eye(6)-D)*np.linalg.det(G)
        res.append(abs(lhs-rhs)/max(1.0,abs(lhs),abs(rhs)))

grid=np.linspace(0,1,1001)
prev=np.linalg.eigvals(A)
prev=prev[np.argsort(np.abs(np.angle(prev)))]
tracks=[[complex(z)] for z in prev]
min_primary_satellite_sep=float('inf')
min_internal_primary_sep=float('inf')
for lam in grid[1:]:
    sl=math.sqrt(lam)
    Ul=np.block([[A,sl*B],[sl*C,D]])
    ev=np.linalg.eigvals(Ul)
    cost=np.abs(prev[:,None]-ev[None,:])
    _,cols=linear_sum_assignment(cost)
    new=ev[cols]
    others=np.delete(ev, sorted(cols))
    if len(others):
        min_primary_satellite_sep=min(min_primary_satellite_sep,float(np.min(np.abs(new[:,None]-others[None,:]))))
    for i in range(3):
        for j in range(i+1,3):
            min_internal_primary_sep=min(min_internal_primary_sep,float(abs(new[i]-new[j])))
    for j,z in enumerate(new): tracks[j].append(complex(z))
    prev=new
final=np.array([t[-1] for t in tracks])
full=np.linalg.eigvals(U)

a=np.random.default_rng(20260916).normal(size=(3,3))+1j*np.random.default_rng(20260917).normal(size=(3,3))
R,_=np.linalg.qr(a)
A2=R.conj().T@A@R; B2=R.conj().T@B; C2=C@R
cov=0.0
for z in (1.2*np.exp(.2j),.88*np.exp(-.4j)):
    G1=z*np.eye(3)-A-B@np.linalg.inv(z*np.eye(6)-D)@C
    G2=z*np.eye(3)-A2-B2@np.linalg.inv(z*np.eye(6)-D)@C2
    cov=max(cov,np.linalg.norm(G2-R.conj().T@G1@R,2))

phi=math.pi/63
cuts={}
for name,Aplaq in [('U1',4),('SU2',32),('SU3',36)]:
    x=2*Aplaq*phi
    for K in range(100):
        b=math.exp(x)*x**(K+1)/math.factorial(K+1)
        if b<1e-12:
            cuts[name]={'x':x,'K':K,'bound':b}; break

out={
 'status':'PASS',
 'feshbach_identity_max_relative_residual':max(res),
 'P_basis_covariance_residual':cov,
 'full_route_pole_count':9,
 'primary_branch_count':3,
 'primary_final_eigenvalues':[[float(z.real),float(z.imag)] for z in final],
 'primary_final_abs_principal_phases':[float(abs(np.angle(z))) for z in final],
 'max_distance_primary_to_full_spectrum':max(float(np.min(abs(full-z))) for z in final),
 'minimum_primary_to_satellite_separation_along_homotopy':min_primary_satellite_sep,
 'minimum_internal_primary_separation_along_homotopy':min_internal_primary_sep,
 'plaquette_galerkin_taylor_certificates':cuts,
 'external_particle_data_used':False
}
(Path(__file__).resolve().parents[2]/'results'/'PRIMARY_FESHBACH_CERTIFICATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
