#!/usr/bin/env python3
"""Exact finite-regulator C8 U(1) static-charge strip diagnostic.

This diagnostic is not an electron propagator. It isolates the compact
U(1) electric-plus-plaquette C8 layer in the Gauss-law sector of a 1 x R strip.
The exact gauge-invariant Hilbert space is reduced to R loop-electric integers.
Odd Z_N regulators N=13,15,17 represent q=-6 without charge aliasing.

For R=1,2,3 the program constructs the Gauss-law cycle basis, the oriented
electric cost P_+(E)=2E^2+E, the reciprocal plaquette cost
4(1-cos theta), the finite-regulator Floquet unitary, a parameter-free positive-
cost reference state for diagnostics only, charged moments, Toeplitz positivity,
and a Hankel matrix-pencil approximation. The positive-cost state is not promoted
to a physical C8 vacuum: C1-C8 do not select it.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import svd

PHI=math.pi/63.0
Q=-6
REGULATORS=(13,15,17)
RVALUES=(1,2,3)
NMOM=64
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)

def P(E): return 2*E*E+E

def graph_matrices(R:int):
    nv=2*(R+1); ne=3*R+1
    B=np.zeros((nv,ne),dtype=int); names=[]; e=0
    top=lambda j:j; bot=lambda j:(R+1)+j
    for j in range(R):
        B[top(j),e]+=1; B[top(j+1),e]-=1; names.append(f't{j}'); e+=1
    for j in range(R):
        B[bot(j),e]+=1; B[bot(j+1),e]-=1; names.append(f'b{j}'); e+=1
    for j in range(R+1):
        B[top(j),e]+=1; B[bot(j),e]-=1; names.append(f'v{j}'); e+=1
    C=np.zeros((ne,R),dtype=int); top0=0; bot0=R; vert0=2*R
    for j in range(R):
        C[top0+j,j]=+1; C[bot0+j,j]=-1; C[vert0+j+1,j]=+1; C[vert0+j,j]=-1
    e0=np.zeros(ne,dtype=int); e0[:R]=Q
    rho=B@e0
    assert np.all(B@C==0)
    return B,C,e0,rho,names

def centered(vals,N):
    vals=np.asarray(vals,dtype=int)%N
    return np.where(vals<=N//2,vals,vals-N)

def configs(R,N):
    rep=centered(np.arange(N),N)
    mesh=np.meshgrid(*([rep]*R),indexing='ij')
    return rep,np.stack([x.reshape(-1) for x in mesh],axis=1)

def electric_diag(R,N,q):
    _,C,e0,rho,names=graph_matrices(R)
    ebase=np.zeros_like(e0); ebase[:R]=q
    _,M=configs(R,N)
    E=ebase[None,:]+M@C.T
    vals=np.sum(P(E),axis=1).astype(float)
    return vals,E,rho,C,names

def theta_B_diag(R,N):
    th=2*np.pi*np.arange(N)/N
    mesh=np.meshgrid(*([th]*R),indexing='ij')
    out=np.zeros([N]*R,float)
    for x in mesh: out += 4*(1-np.cos(x))
    return out

def fft(v,R,N,inverse=False):
    a=np.asarray(v,complex).reshape([N]*R)
    b=np.fft.ifftn(a,norm='ortho') if inverse else np.fft.fftn(a,norm='ortho')
    return b.reshape(-1)

def apply_Bdiag(v,Bdiag,R,N):
    w=fft(v,R,N,False); w=Bdiag.reshape(-1)*w
    return fft(w,R,N,True)

def apply_U(v,KE,Bdiag,R,N):
    w=np.exp(-1j*PHI*KE)*v
    w=fft(w,R,N,False); w=np.exp(-1j*PHI*Bdiag.reshape(-1))*w
    return fft(w,R,N,True)

def ground_ref(R,N):
    KE0,_,_,_,_=electric_diag(R,N,0); Bdiag=theta_B_diag(R,N); dim=N**R
    def mv(v): return KE0*v+apply_Bdiag(v,Bdiag,R,N)
    H=LinearOperator((dim,dim),matvec=mv,rmatvec=mv,dtype=np.complex128)
    v0=np.ones(dim,complex)/math.sqrt(dim)
    vals,vecs=eigsh(H,k=2,which='SA',v0=v0,tol=1e-12,maxiter=30000)
    o=np.argsort(vals); vals=vals[o]; vecs=vecs[:,o]
    v=vecs[:,0]/np.linalg.norm(vecs[:,0])
    return vals,v,Bdiag

def toeplitz_min(c,m=20):
    m=min(m,(len(c)+1)//2); T=np.empty((m,m),complex)
    for i in range(m):
        for j in range(m):
            d=j-i; T[i,j]=c[d] if d>=0 else np.conj(c[-d])
    return float(np.linalg.eigvalsh((T+T.conj().T)/2).min())

def matrix_pencil(c,max_rank=16,tol=1e-9):
    c=np.asarray(c,complex); K=min((len(c)-1)//2,32)
    H0=np.array([[c[i+j] for j in range(K)] for i in range(K)],complex)
    H1=np.array([[c[i+j+1] for j in range(K)] for i in range(K)],complex)
    U,s,Vh=svd(H0,full_matrices=False)
    if s[0]==0:return [],s
    r=max(1,min(max_rank,int(np.sum(s/s[0]>tol))))
    Ur=U[:,:r]; Vr=Vh.conj().T[:,:r]; sr=s[:r]
    A=(Ur.conj().T@H1@Vr)/sr[None,:]
    z=np.linalg.eigvals(A); z=z[np.argsort(np.abs(np.abs(z)-1))][:r]
    V=np.vander(z,N=r,increasing=True).T
    try:w=np.linalg.solve(V,c[:r])
    except np.linalg.LinAlgError:w=np.linalg.lstsq(V,c[:r],rcond=None)[0]
    out=[(float(np.angle(zi)),float(abs(zi)),float(np.real(wi)),float(abs(wi))) for zi,wi in zip(z,w)]
    out.sort(key=lambda x:-x[3])
    return out,s

def run_case(R,N,save_moments=False):
    gvals,gv,Bdiag=ground_ref(R,N)
    KEq,_,rho,C,_=electric_diag(R,N,Q); src=gv.copy(); c=[]; psi=src.copy(); normerr=0.0
    for _ in range(NMOM):
        c.append(np.vdot(src,psi)); normerr=max(normerr,abs(np.linalg.norm(psi)-1)); psi=apply_U(psi,KEq,Bdiag,R,N)
    pencil,s=matrix_pencil(c); dom=pencil[0] if pencil else (float('nan'),)*4
    if save_moments:
        with open(DATA/f'c8_static_charge_moments_R{R}_N{N}.csv','w',newline='') as f:
            w=csv.writer(f);w.writerow(['n','real','imag','abs'])
            for n,z in enumerate(c):w.writerow([n,float(z.real),float(z.imag),float(abs(z))])
    return {
      'R':R,'N':N,'dimension':N**R,'q':Q,'cycle_rank':int(C.shape[1]),'gauss_source':rho.tolist(),
      'reference_ground_energy':float(gvals[0]),'reference_gap':float(gvals[1]-gvals[0]),
      'unitary_norm_max_error':float(normerr),'toeplitz_min_eigenvalue':toeplitz_min(c),
      'c1_abs':float(abs(c[1])),'c2_abs':float(abs(c[2])),'c10_abs':float(abs(c[10])),
      'dominant_pencil_phase':dom[0],'dominant_pencil_radius':dom[1],'dominant_pencil_weight_abs':dom[3],
      'hankel_singular_values_first10':[float(x) for x in s[:10]],'pencil_nodes':pencil,
      'interpretation':'Exact finite-regulator static open-charge gauge diagnostic; not the full C8 electron propagator.'
    }

def main():
    results=[]
    for N in REGULATORS:
        for R in RVALUES:
            r=run_case(R,N,save_moments=(N==13)); results.append(r)
            print(f"PASS-DIAG N={N} R={R} dim={r['dimension']} C1={r['c1_abs']:.12g} norm={r['unitary_norm_max_error']:.3g}")
    fields=['N','R','dimension','q','reference_ground_energy','reference_gap','unitary_norm_max_error','toeplitz_min_eigenvalue','c1_abs','c2_abs','c10_abs','dominant_pencil_phase','dominant_pencil_radius','dominant_pencil_weight_abs']
    with open(DATA/'c8_static_charge_strip_convergence.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in results])
    (DATA/'c8_static_charge_strip_full.json').write_text(json.dumps(results,indent=2)+'\n')
    # Regulator stability of basic diagnostics at fixed R.
    max_c1_spread=max(max(x['c1_abs'] for x in results if x['R']==R)-min(x['c1_abs'] for x in results if x['R']==R) for R in RVALUES)
    max_gap_spread=max(max(x['reference_gap'] for x in results if x['R']==R)-min(x['reference_gap'] for x in results if x['R']==R) for R in RVALUES)
    cert={
      'status':'PASS_DIAGNOSTIC','external_particle_mass_data_used':False,'phi_star_exact':'pi/63','charge_integer':Q,
      'regulators':list(REGULATORS),'R_values':list(RVALUES),'all_norm_errors_below_1e-10':all(r['unitary_norm_max_error']<1e-10 for r in results),
      'all_toeplitz_min_above_minus_1e-9':all(r['toeplitz_min_eigenvalue']>-1e-9 for r in results),
      'max_C1_abs_regulator_spread':float(max_c1_spread),'max_reference_gap_regulator_spread':float(max_gap_spread),
      'physical_electron_pole_claimed':False,
      'reason_not_electron':'The strip retains exact U(1) C8 electric/plaquette dynamics and Gauss law but omits full BCC dynamical matter streaming, the complete receiver/Clebsch tensor, non-Abelian sectors, and a C8-selected thermodynamic vacuum.',
      'reference_state_status':'ground state of K_E+K_B used only as a deterministic numerical diagnostic; it is not selected by C1-C8.'
    }
    (CERT/'c8_static_charge_strip_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    assert cert['all_norm_errors_below_1e-10'] and cert['all_toeplitz_min_above_minus_1e-9']
    print(json.dumps(cert,indent=2))
if __name__=='__main__':main()
