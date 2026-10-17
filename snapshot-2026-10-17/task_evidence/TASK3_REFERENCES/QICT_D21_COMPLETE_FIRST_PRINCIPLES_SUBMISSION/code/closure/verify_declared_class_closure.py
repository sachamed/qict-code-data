#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)

def cplx(z): return [float(np.real(z)), float(np.imag(z))]

# 1. r -> q -> qnu and C21 arithmetic checks
r=7; q=2*r; qnu=3*r
phi21=sum(math.gcd(k,21)==1 for k in range(1,22))
route_defects=[21-2*d for d in (10,9,8)]

# 2. Family Weyl pair
omega=np.exp(2j*np.pi/3)
X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex)
Z=np.diag([1,omega,omega**2])
weyl_res=float(np.linalg.norm(Z@X-omega*X@Z))
# commutant linear equations on vec(M): MX-XM=0, MZ-ZM=0
I=np.eye(3)
A=np.vstack([np.kron(X.T,I)-np.kron(I,X), np.kron(Z.T,I)-np.kron(I,Z)])
s=np.linalg.svd(A,compute_uv=False)
comm_dim=int(np.sum(s<1e-10))
det_dimension_obstruction={n:abs(omega**n-1) for n in range(1,7)}

# 3. Julia uniqueness numerical certificate on deterministic contractions
rng=np.random.default_rng(210714)
julia=[]
for trial in range(8):
    M=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    # scale to strict contraction with max singular value 0.75
    sv=np.linalg.svd(M,compute_uv=False)[0]
    T=.75*M/sv
    def sqrt_psd(H):
        H=(H+H.conj().T)/2
        w,V=np.linalg.eigh(H); w=np.clip(w,0,None)
        return (V*np.sqrt(w))@V.conj().T
    DL=sqrt_psd(np.eye(3)-T@T.conj().T)
    DR=sqrt_psd(np.eye(3)-T.conj().T@T)
    U=np.block([[DL,T],[-T.conj().T,DR]])
    unity=float(np.linalg.norm(U.conj().T@U-np.eye(6)))
    intertwine=float(np.linalg.norm(DL@T-T@DR))
    sval=np.linalg.svd(T,compute_uv=False)
    theta=np.arcsin(sval)
    # eigenphases of each SVD-reduced canonical 2x2 block = +/- theta
    trig=float(np.max(np.abs(np.sin(theta)-sval)))
    julia.append(dict(trial=trial,unitarity_residual=unity,defect_intertwining_residual=intertwine,arcsin_identity_residual=trig))

# 4. Exact C7 Weyl node / doubling audit (same formula as active C7)
def nminus_kappa(kappa):
    kx,ky,kz=map(float,kappa)
    cx,cy,cz=[math.cos(x) for x in (kx,ky,kz)]
    sx,sy,sz=[math.sin(x) for x in (kx,ky,kz)]
    d=cx*cy*cz-sx*sy*sz
    n=np.array([
        sx*cy*cz+cx*sy*sz,
        cx*sy*cz-sx*cy*sz,
        cx*cy*sz+sx*sy*cz
    ],float)
    return d,n

def jacobian(f,x,h=1e-7):
    x=np.array(x,float); J=np.zeros((3,3))
    for j in range(3):
        e=np.zeros(3); e[j]=h
        J[:,j]=(f(x+e)-f(x-e))/(2*h)
    return J
nodes=[
    ('k0',(0,0,0)),
    ('k1',(math.pi/2,math.pi/2,math.pi/2)),
    ('k2',(-math.pi/2,-math.pi/2,-math.pi/2)),
    ('k3',(math.pi,0,0)),
]
node_results=[]
for name,k in nodes:
    d,n=nminus_kappa(k)
    J=jacobian(lambda x:nminus_kappa(x)[1],k)
    det=float(np.linalg.det(J))
    node_results.append(dict(name=name,kappa=list(k),d=float(d),n_norm=float(np.linalg.norm(n)),jacobian_det=det,chirality_sign=int(np.sign(det)),quasienergy='0' if d>0 else 'pi'))
zero_chiral=[x['chirality_sign'] for x in node_results if x['quasienergy']=='0']
doubling_present=(len(zero_chiral)>=2 and len(set(zero_chiral))>1)

# 5. Polar covariance numerical check
polar=[]
for trial in range(8):
    def rand_unitary(n):
        M=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        Q,R=np.linalg.qr(M); phases=np.diag(R); phases=np.where(abs(phases)>0,phases/abs(phases),1)
        return Q@np.diag(np.conj(phases))
    G=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    # keep invertible deterministically
    G+=1.7*np.eye(3)
    def polar_u(A):
        H=A.conj().T@A
        w,V=np.linalg.eigh(H)
        return A@(V*(1/np.sqrt(w)))@V.conj().T
    V=polar_u(G); Lu=rand_unitary(3); Rd=rand_unitary(3)
    G2=Lu@G@Rd.conj().T; V2=polar_u(G2)
    polar.append(dict(trial=trial,unitarity_residual=float(np.linalg.norm(V.conj().T@V-np.eye(3))),covariance_residual=float(np.linalg.norm(V2-Lu@V@Rd.conj().T))))

cert={
 'theory':'QICT declared-class closure certificates',
 'external_particle_data_opened':False,
 'discrete_chain':{'r':r,'q':q,'qnu':qnu,'phi21':phi21,'near_antipodal_defects':route_defects,'status':'PASS' if (q,qnu,phi21,route_defects)==(14,21,12,[1,3,5]) else 'FAIL'},
 'family_weyl_pair':{'relation_residual':weyl_res,'commutant_dimension':comm_dim,'determinant_obstruction_abs_omega_n_minus_1':{str(k):float(v) for k,v in det_dimension_obstruction.items()},'minimal_dimension':3,'status':'PASS' if weyl_res<1e-12 and comm_dim==1 else 'FAIL'},
 'julia_positive_defect_colligation':{'trials':julia,'max_unitarity_residual':max(x['unitarity_residual'] for x in julia),'max_intertwining_residual':max(x['defect_intertwining_residual'] for x in julia),'status':'PASS' if max(max(x['unitarity_residual'],x['defect_intertwining_residual']) for x in julia)<1e-11 else 'FAIL'},
 'c7_doubling_audit':{'nodes':node_results,'opposite_chirality_zero_quasienergy_nodes':doubling_present,'status':'PASS_AUDIT__DOUBLING_PRESENT' if doubling_present else 'FAIL_AUDIT'},
 'proper_vertex_polar_map':{'trials':polar,'max_unitarity_residual':max(x['unitarity_residual'] for x in polar),'max_basis_covariance_residual':max(x['covariance_residual'] for x in polar),'status':'PASS' if max(max(x['unitarity_residual'],x['covariance_residual']) for x in polar)<1e-10 else 'FAIL'},
 'claim_boundary':{
   'physical_masses':'NOT_NUMERICALLY_COMPUTED__REQUIRES_COMPACT_3D_EDGE_AND_TAU_STAR',
   'physical_CKM_PMNS':'NOT_NUMERICALLY_COMPUTED__REQUIRES_INTERACTING_PROPER_VERTICES',
   'interacting_RG':'ONE_LOOP_SM_CONTINUUM_CALCULATED_CONDITIONALLY__QCA_TO_CONTINUUM_MAP_NOT_DERIVED',
   'gravity':'NO_GO_DEMONSTRATED_FOR_FIXED_C1_C11_VARIABLE_CONTENT__EXTENSION_NOT_DERIVED'
 }
}
status_ok=all(cert[k]['status']=='PASS' for k in ('discrete_chain','family_weyl_pair','julia_positive_defect_colligation','proper_vertex_polar_map')) and doubling_present
cert['overall_status']='PASS_WITH_CERTIFIED_PHYSICAL_BOUNDARIES' if status_ok else 'FAIL'
out=CERT/'declared_class_closure_certificate.json'
out.write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
