#!/usr/bin/env python3
"""Deterministic checks of spectral response and regulator error bounds.

The mathematical assumptions are stated in Supplement S105. The checks
evaluate finite-dimensional identities and error bounds.
"""
from pathlib import Path
import json, subprocess, sys, tempfile
import numpy as np
from scipy.linalg import expm
from execute_physical_g2_contract_audit import input_status, REQ

ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(18052026)
checks={}
def record(name,ok,**data):
    checks[name]={'pass':bool(ok),**data}
    if not ok: raise AssertionError(name)

def herm(n):
    a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    return (a+a.conj().T)/2

# Generic real mixed derivative, including noncommuting sources.
K=np.diag([0.,2.,5.]).astype(complex); A=herm(3)/4; B=herm(3)/4; D=herm(3)/4
P=np.diag([1.,0.,0.]); S=np.diag([0.,-.5,-.2])
analytic=float(np.real(np.trace(P@(D+A@S@B+B@S@A))))
def ev(x,y):return np.linalg.eigvalsh(K+x*A+y*B+x*y*D)[0]
h=1e-4
finite=(ev(h,h)-ev(h,-h)-ev(-h,h)+ev(-h,-h))/(4*h*h)
record('mixed_real_kato',abs(analytic-finite)<2e-7,analytic=analytic,finite_difference=float(finite),absolute_error=abs(analytic-finite))

with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'input.npz'; out=Path(tmp)/'out.json'
    plus=np.array([[0.,1.],[0.,0.]],complex)
    payload=dict(K0=np.diag([0.,2.]),K1_vol=np.eye(2),K1_q=plus,K1_mq=plus.conj().T,
                 K2_qmq=np.zeros((2,2)),N_cells=1,qell=.2,continuum_density_factor=1.,source_basis='fourier_pair')
    def run_payload(p):
        np.savez(path,**p)
        return subprocess.run([sys.executable,str(ROOT/'code/compute_spectral_gravity_response.py'),str(path),'-o',str(out)],capture_output=True,text=True)
    result=run_payload(payload)
    if result.returncode:raise AssertionError(result.stderr+result.stdout)
    response=json.loads(out.read_text())['chi_hat_TT_R_q']
    record('fourier_mixed_response_not_doubled',abs(response+.5)<1e-13,response=response,incorrect_doubled_value=-1.)
    bad=dict(payload);bad['K0']=np.array([[0.,1.],[0.,2.]])
    result=run_payload(bad)
    record('nonhermitian_input_rejected',result.returncode!=0 and 'Hermitian' in result.stderr)
    bad=dict(payload);bad['K1_mq']=plus
    result=run_payload(bad)
    record('invalid_fourier_pair_rejected',result.returncode!=0 and 'Fourier' in result.stderr)
    bad=dict(payload);bad['K0']=np.zeros((2,2))
    result=run_payload(bad)
    record('unseparated_branch_rejected',result.returncode!=0 and 'separated' in result.stderr)

# Receiver/update error bound, not just update error on exact source columns.
n,d,m=30,6,8
U=expm(1j*herm(n));V=np.linalg.qr(rng.normal(size=(n,d))+1j*rng.normal(size=(n,d)))[0]
E=herm(n);E*=2e-6/np.linalg.norm(E,2)
F=rng.normal(size=(n,d))+1j*rng.normal(size=(n,d));F*=3e-6/np.linalg.norm(F,2)
Uh,Vh=U+E,V+F;delta=float(np.linalg.norm(E,2));nu=float(np.linalg.norm(F,2))
C=[];Ch=[];eta=[];ratios=[]
for k in range(m+1):
    c=V.conj().T@np.linalg.matrix_power(U,k)@V; ch=Vh.conj().T@np.linalg.matrix_power(Uh,k)@Vh
    bound=(1+nu)**2*np.expm1(k*np.log1p(delta))+2*nu+nu*nu
    C.append(c);Ch.append(ch);eta.append(bound);ratios.append(float(np.linalg.norm(ch-c,2)/bound))
record('joint_receiver_update_bound',max(ratios)<=1+1e-8,max_error_to_bound_ratio=max(ratios))
T=np.block([[Ch[j-k] if j>=k else Ch[k-j].conj().T for k in range(m+1)] for j in range(m+1)])
floor=-eta[0]-2*sum(eta[1:]);mineig=float(np.linalg.eigvalsh(T)[0])
record('toeplitz_error_envelope',mineig>=floor-1e-12,minimum_eigenvalue=mineig,lower_envelope=float(floor))

# An arbitrarily small low-energy atom changes the hard edge.
eps=1e-9
errors=[abs((1-eps)*1.**k+eps*0.**k-1.) for k in range(1,100)]
record('weak_moments_do_not_certify_edge',max(errors)<2e-9,moment_error=max(errors),finite_edge=0,limit_edge=1)

# Matrix rank and parameter-space dimension are different.
v=np.array([-1/np.sqrt(6),1/np.sqrt(3),-1/np.sqrt(2)])
D=np.diag([1.,2.,4.]);B=np.outer(v,v)@D+D@np.outer(v,v)
rank=int(np.linalg.matrix_rank(B));deformation_rank=int(np.linalg.matrix_rank(np.stack([B.ravel(),2*B.ravel(),3*B.ravel()])))
record('radiative_rank_vs_parameter_dimension',rank==2 and deformation_rank==1,matrix_rank=rank,parameter_span_rank=deformation_rank)

J=np.array([[.25,.25],[0.,.25]]);k=12
scaled=np.linalg.matrix_power(J,k)*4.**k
record('peripheral_jordan_logarithm',np.linalg.norm(scaled-np.array([[1.,k],[0.,1.]]))<1e-12,scaled_norm=float(np.linalg.norm(scaled,2)),iteration=k)

for flags,name in [({},'empty'),({key:True for key in REQ},'all_true')]:
    status=input_status(flags)
    record('readiness_'+name+'_cannot_certify_continuum',not status['physical_G2_promoted'],input_flags_complete=status['input_flags_complete'])

# Independence and the statistical status are executable conditions.
fit=json.loads((ROOT/'results/phenomenology/PHENOMENOLOGY_CHI2_AUDIT.json').read_text())
check=fit['conditional_reference_check']; datum=check['observables'][0]
value=((datum['conditional_prediction']-datum['measurement'])/datum['sigma'])**2
record('conditional_LEP_not_global_fit',abs(value-.25)<1e-12 and fit['global_fit']['N_independent_QICT_continuum_predictions']==0 and not check['independent_QICT_prediction'],chi2=value)


# Finite-time analytic source bound with noncommuting baseline/source matrices.
dim=4
base=[herm(dim)/3 for _ in range(3)]
bp=[herm(dim)/8 for _ in base]; bm=[herm(dim)/9 for _ in base]
strength=sum(np.linalg.norm(x,2) for x in bp+bm)
a0=rng.normal(size=(dim,dim))+1j*rng.normal(size=(dim,dim));rho=a0@a0.conj().T;rho/=np.trace(rho)
def branch(z,bs):
    v=np.eye(dim,dtype=complex)
    for a,b in zip(base,bs):v=expm(-1j*(a+z*b))@v
    return v
def source(z):return np.trace(rho@np.linalg.solve(branch(z,bm),branch(z,bp)))
rr=.75*np.log(2)/strength
points=[rr*np.exp(2j*np.pi*j/64) for j in range(64)]
ratios=[abs(source(z)-1)/np.expm1(strength*abs(z)) for z in points]
record('bounded_source_common_disk',max(ratios)<=1+1e-12 and min(abs(source(z)) for z in points)>0,radius=float(rr),integrated_norm=float(strength),max_error_to_bound_ratio=float(max(ratios)))
# Cauchy's derivative independently agrees with the real finite difference.
first=sum(source(z)/z for z in points)/len(points)
h=1e-5;difference=(source(h)-source(-h))/(2*h)
record('source_cauchy_derivative',abs(first-difference)<1e-8,absolute_error=float(abs(first-difference)))
# Full three-mode Fock implementation, built from minors of the one-particle unitary.
def lift(u):
    n=u.shape[0];g=np.zeros((2**n,2**n),complex)
    for i in range(2**n):
        a=[k for k in range(n) if i>>k&1]
        for j in range(2**n):
            b=[k for k in range(n) if j>>k&1]
            if len(a)==len(b):g[i,j]=np.linalg.det(u[np.ix_(a,b)]) if a else 1
    return g
u=expm(1j*herm(3));v=expm(1j*herm(3));g=lift(u);creators=[]
for mode in range(3):
    c=np.zeros((8,8),complex)
    for j in range(8):
        if not(j>>mode&1):c[j|(1<<mode),j]=(-1)**sum((j>>k)&1 for k in range(mode))
    creators.append(c)
residual=max(np.linalg.norm(g@creators[j]@g.conj().T-sum(u[i,j]*creators[i] for i in range(3)),2) for j in range(3))
composition=np.linalg.norm(lift(u@v)-g@lift(v),2)
record('vacuum_CAR_implies_Fock_lift',residual<1e-12 and composition<1e-12 and abs(g[0,0]-1)<1e-15,intertwining_error=float(residual),composition_error=float(composition))

report={'classification':'FINITE_RESPONSE_BOUNDS_CHECKS','scope':'Finite-dimensional examples and consistency checks for analytic response bounds.','checks':checks,'pass':all(c['pass'] for c in checks.values()),'check_count':len(checks)}
path=ROOT/'results/RESPONSE_BOUNDS_VALIDATION.json'
path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(f'PASS {len(checks)} finite-response consistency checks')
