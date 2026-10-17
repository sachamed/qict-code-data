import sys, math, json, csv, time
from pathlib import Path
import numpy as np
from scipy.linalg import schur

PKG=Path('/mnt/data/QICT_D21_ROUTE_FIRST_GAUGE_CLOSURE_SUBMISSION_2026-09-15')
sys.path.insert(0,str(PKG/'code'))
sys.path.insert(0,str(PKG/'code'/'vendor'))
from qict_route_first import electron_route_coefficients, I3, X, Z
import compute_c8_static_charge_strip as strip
PHI=math.pi/63
Wfam=[I3,X,Z]
coeff=electron_route_coefficients()

def dft_matrix(N):
    j=np.arange(N)
    return np.exp(-2j*np.pi*np.outer(j,j)/N)/np.sqrt(N)

def dense_components(R,N,q):
    KE,_,_,_,_=strip.electric_diag(R,N,q)
    B=strip.theta_B_diag(R,N).reshape(-1)
    F=dft_matrix(N)
    FR=F
    for _ in range(R-1): FR=np.kron(FR,F)
    return KE,B,FR

def Uof(KE,B,F,s=1.0):
    DE=np.exp(-1j*s*PHI*KE); DB=np.exp(-1j*s*PHI*B)
    return F.conj().T @ (DB[:,None]*(F*DE[None,:]))

def kato_vacuum(R,N,steps=41):
    vals,g,_=strip.ground_ref(R,N); g=g/np.linalg.norm(g)
    KE,B,F=dense_components(R,N,0)
    prev=g.copy(); chosen=None; lamj=None; mingap=99.; minov=1.
    for s in np.linspace(0.02,1.0,steps):
        U=Uof(KE,B,F,float(s)); T,Q=schur(U,output='complex'); lam=np.diag(T)
        ovs=np.abs(Q.conj().T@prev)**2; j=int(np.argmax(ovs)); v=Q[:,j]
        al=np.vdot(prev,v)
        if abs(al)>0: v*=np.exp(-1j*np.angle(al))
        phase=float(((-np.angle(lam[j]))+np.pi)%(2*np.pi)-np.pi)
        allph=(((-np.angle(lam))+np.pi)%(2*np.pi)-np.pi)
        dif=np.abs(((allph-phase)+np.pi)%(2*np.pi)-np.pi); dif[j]=99
        mingap=min(mingap,float(np.min(dif))); minov=min(minov,float(ovs[j]))
        prev=v; chosen=U; lamj=lam[j]
    lamj=np.vdot(prev,chosen@prev); lamj/=abs(lamj)
    return prev,lamj,{'primitive_E0':float(vals[0]),'min_step_overlap':minov,'min_gap':mingap,'final_overlap_primitive':float(abs(np.vdot(g,prev))**2)}

def family_atoms(W):
    T,Q=schur(W,output='complex'); lam=np.diag(T)
    out=[]; used=np.zeros(3,bool)
    for i in range(3):
        if used[i]: continue
        idx=np.where(np.abs(lam-lam[i])<1e-10)[0]; used[idx]=True
        for jj in idx:
            u=Q[:,jj]
            out.append((lam[i],u))
    return out

def build_measure(N,R=2):
    d=N*N
    vac,lam0,kinfo=kato_vacuum(R,N)
    KE,B,F=dense_components(R,N,-6); Uq=Uof(KE,B,F,1.0)
    S=np.roll(np.eye(N,dtype=complex),1,axis=0)
    hol=[np.eye(d,dtype=complex),np.kron(S,np.eye(N)),np.kron(np.eye(N),S)]
    xs=[]; rows=[]; max_unit=0
    for a in range(3):
        G=hol[a]@Uq
        max_unit=max(max_unit,float(np.linalg.norm(G.conj().T@G-np.eye(d),2)))
        Tg,Qg=schur(G,output='complex'); lg=np.diag(Tg)
        wg=np.abs(Qg.conj().T@vac)**2
        for lf,u in family_atoms(Wfam[a]):
            for k in range(d):
                w=float(wg[k]/3.0)
                if w < 1e-18: continue
                z=coeff[a]*lf*lg[k]*np.conj(lam0)
                # Cayley: x = i(1-z)/(1+z), real for |z|=1
                den=1+z
                if abs(den)<1e-12: continue
                x=(1j*(1-z)/den).real
                xs.append(float(x))
                rows.append(np.sqrt(w)*np.conj(u))  # row so row^H row = w u u^H
    xs=np.array(xs,float)
    V=np.array(rows,complex)
    gram=V.conj().T@V
    return xs,V,gram,kinfo,max_unit

def block_lanczos_diag(xs,V,nblocks=7,tol=1e-11):
    # Orthonormalize initial block while preserving source basis covariance.
    Q,R=np.linalg.qr(V)
    # Source Gram ideally I; R retained only as certificate.
    Qs=[Q[:,:3]]; As=[]; Bs=[]
    Qprev=None; Bprev=None
    for n in range(nblocks):
        Qn=Qs[-1]
        KQ=xs[:,None]*Qn
        A=Qn.conj().T@KQ; A=(A+A.conj().T)/2
        As.append(A)
        Rr=KQ-Qn@A
        if n>0:
            Rr-=Qprev@Bprev.conj().T
        for Qo in Qs:
            Rr-=Qo@(Qo.conj().T@Rr)
        # QR; canonical positive B via polar rotation of next Q
        qnext,r=np.linalg.qr(Rr)
        s=np.linalg.svd(r,compute_uv=False)
        if np.min(s)<tol:
            break
        # Keep QR convention; singular values are gauge-invariant.
        B=qnext[:,:3].conj().T@Rr
        Bs.append(B)
        Qprev=Qn; Bprev=B; Qs.append(qnext[:,:3])
    return As,Bs,R

def invs(As,Bs):
    return {
      'A_eigs':[np.linalg.eigvalsh(A).tolist() for A in As],
      'B_svals':[np.sort(np.linalg.svd(B,compute_uv=False)).tolist() for B in Bs],
      'A_frob':[float(np.linalg.norm(A,'fro')) for A in As],
      'B_frob':[float(np.linalg.norm(B,'fro')) for B in Bs],
    }

def run(N):
    t=time.time(); xs,V,gram,kinfo,udef=build_measure(N)
    As,Bs,R=block_lanczos_diag(xs,V,7)
    return {'N':N,'states':len(xs),'seconds':time.time()-t,
            'source_gram':[[[float(x.real),float(x.imag)] for x in row] for row in gram],
            'source_gram_defect':float(np.linalg.norm(gram-np.eye(3),2)),
            'initial_qr_R':[[[float(x.real),float(x.imag)] for x in row] for row in R],
            'kato':kinfo,'max_block_unitarity_defect':udef,
            'n_A':len(As),'n_B':len(Bs),'invariants':invs(As,Bs)}

if __name__=='__main__':
    out=[]
    for N in (15,17,19):
        print('RUN',N,flush=True); r=run(N); out.append(r); print('DONE',N,r['seconds'],r['source_gram_defect'],flush=True)
    Path('/mnt/data/BLIND_BLOCKS_15_17_19.json').write_text(json.dumps({'external_mass_input':False,'cases':out},indent=2)+'\n')
    # Flatten CSV invariants.
    rows=[]
    for c in out:
      for n,e in enumerate(c['invariants']['A_eigs']):
        rows.append({'N':c['N'],'kind':'A','shell':n,'v1':e[0],'v2':e[1],'v3':e[2]})
      for n,s in enumerate(c['invariants']['B_svals']):
        rows.append({'N':c['N'],'kind':'B','shell':n,'v1':s[0],'v2':s[1],'v3':s[2]})
    with open('/mnt/data/BLIND_BLOCKS_15_17_19.csv','w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps(out,indent=2)[:8000])
