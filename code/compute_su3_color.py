#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,hashlib
from pathlib import Path
import mpmath as mp
import numpy as np

def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parameters',required=True); ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); par_path=Path(a.parameters); par=json.loads(par_path.read_text()); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    mp.mp.dps=90; I=1j
    lam=np.stack([
      np.array([[0,1,0],[1,0,0],[0,0,0]],complex),
      np.array([[0,-I,0],[I,0,0],[0,0,0]],complex),
      np.array([[1,0,0],[0,-1,0],[0,0,0]],complex),
      np.array([[0,0,1],[0,0,0],[1,0,0]],complex),
      np.array([[0,0,-I],[0,0,0],[I,0,0]],complex),
      np.array([[0,0,0],[0,0,1],[0,1,0]],complex),
      np.array([[0,0,0],[0,0,-I],[0,I,0]],complex),
      np.array([[1,0,0],[0,1,0],[0,0,-2]],complex)/math.sqrt(3)])
    T=lam/2; f=np.zeros((8,8,8)); d=np.zeros((8,8,8)); I3=np.eye(3,dtype=complex)
    for aa in range(8):
      for bb in range(8):
        comm=lam[aa]@lam[bb]-lam[bb]@lam[aa]; anti=lam[aa]@lam[bb]+lam[bb]@lam[aa]
        for cc in range(8):
          f[aa,bb,cc]=float(np.real(np.trace(comm@lam[cc])/(4j)))
          d[aa,bb,cc]=float(np.real(np.trace(anti@lam[cc])/4))
    f[np.abs(f)<1e-14]=0; d[np.abs(d)<1e-14]=0
    vecI=I3.reshape(-1)/math.sqrt(3); P1=np.outer(vecI,vecI.conj()); P8=np.eye(9)-P1
    Swap=np.zeros((9,9),complex)
    for x in range(3):
      for y in range(3): Swap[y*3+x,x*3+y]=1
    P6=(np.eye(9)+Swap)/2; P3b=(np.eye(9)-Swap)/2
    CG1=I3[None,:,:]/math.sqrt(3); CG8=lam/math.sqrt(2)
    eps=np.zeros((3,3,3),complex)
    for x,y,z,s in [(0,1,2,1),(1,2,0,1),(2,0,1,1),(1,0,2,-1),(2,1,0,-1),(0,2,1,-1)]: eps[x,y,z]=s
    CG3b=eps/math.sqrt(2)
    CG6=np.zeros((6,3,3),complex); CG6[0,0,0]=CG6[1,1,1]=CG6[2,2,2]=1
    CG6[3,0,1]=CG6[3,1,0]=CG6[4,0,2]=CG6[4,2,0]=CG6[5,1,2]=CG6[5,2,1]=1/math.sqrt(2)
    def gram(C): V=C.reshape(C.shape[0],-1); return V@V.conj().T
    trace_gram=np.einsum('aij,bji->ab',lam,lam).real
    comm_err=anti_err=0.0
    for aa in range(8):
      for bb in range(8):
        comm_err=max(comm_err,float(np.linalg.norm(lam[aa]@lam[bb]-lam[bb]@lam[aa]-2j*sum(f[aa,bb,cc]*lam[cc] for cc in range(8)),'fro')))
        anti_err=max(anti_err,float(np.linalg.norm(lam[aa]@lam[bb]+lam[bb]@lam[aa]-(4/3)*(aa==bb)*I3-2*sum(d[aa,bb,cc]*lam[cc] for cc in range(8)),'fro')))
    CF=sum(T[aa]@T[aa] for aa in range(8)); ff=np.einsum('acd,bcd->ab',f,f); dd=np.einsum('acd,bcd->ab',d,d)
    comp_err=0.0
    for ii in range(3):
      for jj in range(3):
        for kk in range(3):
          for ll in range(3):
            lhs=sum(T[a0,ii,jj]*T[a0,kk,ll] for a0 in range(8)); rhs=.5*((ii==ll and jj==kk)-(1/3)*(ii==jj and kk==ll)); comp_err=max(comp_err,abs(lhs-rhs))
    V1=CG1.reshape(1,9); V8=CG8.reshape(8,9); V3b=CG3b.reshape(3,9); V6=CG6.reshape(6,9)
    G33b=[np.kron(T[a0],I3)-np.kron(I3,T[a0].conj()) for a0 in range(8)]; C33b=sum(G@G for G in G33b)
    G33=[np.kron(T[a0],I3)+np.kron(I3,T[a0]) for a0 in range(8)]; C33=sum(G@G for G in G33)
    jac=0.0
    for aa in range(8):
      for bb in range(8):
        for cc in range(8):
          for ee in range(8): jac=max(jac,abs(sum(f[aa,bb,x]*f[x,cc,ee]+f[bb,cc,x]*f[x,aa,ee]+f[cc,aa,x]*f[x,bb,ee] for x in range(8))))
    ident={
      'trace_gram_fro_error':float(np.linalg.norm(trace_gram-2*np.eye(8),'fro')),
      'commutator_max_fro_error':comm_err,'anticommutator_max_fro_error':anti_err,
      'fundamental_casimir_fro_error':float(np.linalg.norm(CF-(4/3)*I3,'fro')),
      'f_contraction_fro_error':float(np.linalg.norm(ff-3*np.eye(8),'fro')),
      'd_contraction_fro_error':float(np.linalg.norm(dd-(5/3)*np.eye(8),'fro')),
      'fundamental_completeness_max_abs_error':float(comp_err),
      'CG_3x3bar_to_1_gram_error':float(np.linalg.norm(gram(CG1)-np.eye(1),'fro')),
      'CG_3x3bar_to_8_gram_error':float(np.linalg.norm(gram(CG8)-np.eye(8),'fro')),
      'CG_3x3_to_3bar_gram_error':float(np.linalg.norm(gram(CG3b)-np.eye(3),'fro')),
      'CG_3x3_to_6_gram_error':float(np.linalg.norm(gram(CG6)-np.eye(6),'fro')),
      'CG_3x3bar_completeness_error':float(np.linalg.norm(V1.conj().T@V1+V8.conj().T@V8-np.eye(9),'fro')),
      'CG_3x3_completeness_error':float(np.linalg.norm(V3b.conj().T@V3b+V6.conj().T@V6-np.eye(9),'fro')),
      'C2_singlet_error':float(np.linalg.norm(P1@C33b@P1,'fro')),
      'C2_octet_error':float(np.linalg.norm(P8@C33b@P8-3*P8,'fro')),
      'C2_symmetric6_error':float(np.linalg.norm(P6@C33@P6-(10/3)*P6,'fro')),
      'C2_antisymmetric3bar_error':float(np.linalg.norm(P3b@C33@P3b-(4/3)*P3b,'fro')),
      'jacobi_max_abs_error':float(jac)}
    np.savez_compressed(out/'SU3_LOCAL_TENSORS.npz',gell_mann=lam,generators=T,f_abc=f,d_abc=d,P_singlet=P1,P_octet=P8,P_6=P6,P_3bar=P3b,CG_3x3bar_to_1=CG1,CG_3x3bar_to_8=CG8,CG_3x3_to_3bar=CG3b,CG_3x3_to_6=CG6)

    order=int(par['receiver_central_order']); phi=2*mp.pi/order
    mult=par['compact_gauge_action_multipliers']; x1=mp.mpf(mult['U1'])*phi; x2=mp.mpf(mult['SU2'])*phi; x3=mp.mpf(mult['SU3'])*phi
    def tail(x,N): return mp.e**x-mp.fsum([x**k/mp.factorial(k) for k in range(N+1)])
    def tfrom(x,n): return mp.e**x if n<=0 else tail(x,n-1)
    def u1(N): return 2*mp.e**(x1*x1/4)*tail(x1/2,N)
    def su2(J2):
      y=x2/2; d0=J2+2; return (2/x2)*mp.e**(x2*x2/4)*(y*y*tfrom(y,d0-2)+y*tfrom(y,d0-1))
    def su3(R): return tail(x3,R)
    def propagated(L,n,a0,b0,c0,color=True):
      P=3*L**3; d1=u1(a0); d2=su2(b0); d3=su3(c0) if color else mp.mpf('0')
      dloc=(1+d1)*(1+d2)*(1+d3)-1; one=mp.expm1(P*mp.log1p(dloc)); return d1,d2,d3,mp.expm1(n*mp.log1p(one))
    cut=par['representation_cutoffs']; L=int(par['lattice_L']); n=int(par['moment_depth'])
    d1,d2,d3,alltail=propagated(L,n,int(cut['U1_order']),int(cut['SU2_twice_j']),int(cut['SU3_p_plus_q']),True)
    _,_,_,leptail=propagated(L,n,int(cut['U1_order']),int(cut['SU2_twice_j']),int(cut['SU3_p_plus_q']),False)
    rows=[]
    for R in range(max(0,int(cut['SU3_p_plus_q'])-5),int(cut['SU3_p_plus_q'])+6):
        rows.append({'R':R,'su3_tail':mp.nstr(su3(R),35)})
    with (out/'SU3_REPRESENTATION_TAILS.csv').open('w',newline='') as fh:
      w=csv.DictWriter(fh,fieldnames=['R','su3_tail']); w.writeheader(); w.writerows(rows)

    # Exact common-color cancellation numerical identity test on deterministic Hermitian matrices.
    def unitary(H,s): vals,vec=np.linalg.eigh(H); return (vec*np.exp(1j*s*vals))@vec.conj().T
    Uc=unitary(lam[0]+.37*lam[3]-.21*lam[7],.47)
    H0=np.array([[0,.2,.1j,0],[.2,.1,0,.05j],[-.1j,0,-.2,.15],[0,-.05j,.15,.3]],complex)
    He=np.array([[.05,.1j,.07,0],[-.1j,.2,.04,.03],[.07,.04,-.1,.11j],[0,.03,-.11j,.22]],complex)
    U0e=unitary(H0,.39); Uee=unitary(He,.39); Wfull=np.kron(U0e,Uc).conj().T@np.kron(Uee,Uc); Wew=U0e.conj().T@Uee
    op_err=float(np.linalg.norm(Wfull-np.kron(Wew,I3),'fro'))
    summary={
      'parameters_sha256':sha256(par_path),'tensor_identity_max_error':max(ident.values()),
      'phi_star':mp.nstr(phi,40),'x_U1':mp.nstr(x1,40),'x_SU2':mp.nstr(x2,40),'x_SU3':mp.nstr(x3,40),
      'L_n':{'L':L,'n':n},'cutoffs':cut,
      'L24_n9':{'U1_local_tail':float(d1),'SU2_local_tail':float(d2),'SU3_local_tail':float(d3),
      'all_gauge_propagated_representation_bound':float(alltail),'charged_lepton_relative_bound_after_exact_color_cancellation':float(leptail),
      'color_contribution_removed':float(alltail-leptail)},'common_color_relative_operator_validation_fro_error':op_err}
    (out/'SU3_VALIDATION.json').write_text(json.dumps({'identities':ident,'summary':summary},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
