from __future__ import annotations
import math,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,identity
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm,schur,eigh
PHI=math.pi/63.0
OUT=Path('/mnt/data/task2_continue/TASK2_SU2_FULL_LINK_MATTER_ROUTE.json')

def basis(nmax):
    st=[]; idx={}
    for n in range(nmax+1):
        for m in range(-n,n+1,2):
            for r in range(-n,n+1,2):
                idx[(n,m,r)]=len(st); st.append((n,m,r))
    return st,idx

def cg(n,m,a,N):
    # C(j,m/2;1/2,a/2 | J=N/2,M=(m+a)/2), Condon-Shortley convention.
    den=2*(n+1)
    if N==n+1:
        x=(n+a*m+2)/den
        return math.sqrt(max(0.0,x))
    if N==n-1 and n>0:
        x=(n-a*m)/den
        return (-a)*math.sqrt(max(0.0,x))
    return 0.0

def operators(nmax):
    st,idx=basis(nmax); G=len(st)
    # Fundamental parallel transporter W on gauge x matter-spin (b -> a).
    wr=[];wc=[];wd=[]
    # Character multiplication Mchi on gauge.
    mr=[];mc=[];md=[]
    for i,(n,m,r) in enumerate(st):
        dj=n+1
        for N in (n-1,n+1):
            if N<0 or N>nmax: continue
            dJ=N+1; norm=math.sqrt(dj/dJ)
            # W matrix elements for matter input b and output a.
            for b in (-1,1):
                for a in (-1,1):
                    cm=cg(n,m,a,N); cr=cg(n,r,b,N)
                    if cm==0 or cr==0: continue
                    M=m+a; R=r+b
                    j=idx.get((N,M,R))
                    if j is None: continue
                    row=2*j+(0 if a==-1 else 1); col=2*i+(0 if b==-1 else 1)
                    wr.append(row);wc.append(col);wd.append(norm*cm*cr)
            # character trace sum a=b
            val_by_out={}
            for a in (-1,1):
                cm=cg(n,m,a,N); cr=cg(n,r,a,N)
                if cm==0 or cr==0: continue
                M=m+a;R=r+a;j=idx.get((N,M,R))
                if j is not None: val_by_out[j]=val_by_out.get(j,0.0)+norm*cm*cr
            for j,val in val_by_out.items():
                if abs(val)>0: mr.append(j);mc.append(i);md.append(val)
    W=coo_matrix((wd,(wr,wc)),shape=(2*G,2*G),dtype=complex).tocsr()
    Mchi=coo_matrix((md,(mr,mc)),shape=(G,G),dtype=complex).tocsr()
    ke=np.array([n*(n+2) for n,_,_ in st],float)
    KB=32*identity(G,format='csr',dtype=complex)-16*Mchi
    return st,idx,W,Mchi,ke,KB

def class_kato(nmax,steps=81):
    n=np.arange(nmax+1,dtype=float)
    KE=np.diag(n*(n+2.0)); KB=np.diag(np.full(nmax+1,32.0))
    for i in range(nmax): KB[i,i+1]=KB[i+1,i]=-16.0
    def U(s): return expm(-1j*s*PHI*KB) @ np.diag(np.exp(-1j*s*PHI*np.diag(KE)))
    _,V=eigh(KE+KB); prev=V[:,0].astype(complex); minov=1.;mingap=99.
    for s in np.linspace(.01,1,steps):
        UU=U(float(s));T,Q=schur(UU,output='complex');lam=np.diag(T);ov=np.abs(Q.conj().T@prev)**2;j=int(np.argmax(ov));v=Q[:,j]
        a=np.vdot(prev,v)
        if abs(a):v*=np.exp(-1j*np.angle(a))
        ph=-np.angle(lam);dp=np.abs(((ph-ph[j])+np.pi)%(2*np.pi)-np.pi);dp[j]=99
        minov=min(minov,float(ov[j]));mingap=min(mingap,float(dp.min()));prev=v
    UU=U(1.0); lam=np.vdot(prev,UU@prev);res=np.linalg.norm(UU@prev-lam*prev)
    return prev,complex(lam),minov,mingap,float(res)

def embed_class(v,st,idx):
    g=np.zeros(len(st),complex)
    for n,amp in enumerate(v):
        d=n+1
        for m in range(-n,n+1,2): g[idx[(n,m,m)]] += amp/math.sqrt(d)
    return g

def run(nmax):
    t=time.time();st,idx,W,Mchi,ke,KB=operators(nmax);G=len(st)
    v,lam,minov,mingap,eigres=class_kato(nmax)
    om=embed_class(v,st,idx); om/=np.linalg.norm(om)
    # Validate Mchi on class subspace: should reproduce adjacency.
    adj_res=0.0
    for n in range(min(nmax,6)+1):
        e=np.zeros(nmax+1);e[n]=1; ge=embed_class(e,st,idx); lhs=Mchi@ge
        tgt=np.zeros(G,complex)
        if n>0:
            e2=np.zeros(nmax+1);e2[n-1]=1;tgt+=embed_class(e2,st,idx)
        if n<nmax:
            e2=np.zeros(nmax+1);e2[n+1]=1;tgt+=embed_class(e2,st,idx)
        adj_res=max(adj_res,float(np.linalg.norm(lhs-tgt)))
    # W is unitary on columns whose gauge n <= nmax-1; evaluate basis random-ish all interior via Gram diagonal/block.
    interior=[]
    for i,(n,m,r) in enumerate(st):
        if n<nmax:
            interior.extend([2*i,2*i+1])
    # Sparse Gram on interior; dimension manageable.
    Wi=W[:,interior]; Gram=(Wi.conj().T@Wi).tocsr(); diff=Gram-identity(len(interior),format='csr',dtype=complex)
    winter=float(np.linalg.norm(diff.data)) if diff.nnz else 0.0
    ephase=np.exp(-1j*PHI*ke)
    def gauge_apply(X):
        # X G x 2; electric rightmost then magnetic.
        Y=ephase[:,None]*X
        return expm_multiply((-1j*PHI)*KB,Y)
    routes=[]
    # initial matter basis average
    for d in (8,9,10):
        vals=[]
        for b in range(2):
            X=np.zeros((G,2),complex); X[:,b]=om
            for _ in range(d):
                # canonical simplified C8 ordering: stream W first, then electric+magnetic gauge step
                X=(W@X.reshape(-1)).reshape(G,2)
                X=gauge_apply(X)
            vals.append(np.vdot(om,X[:,b]))
        amp=sum(vals)/2/(lam**d)
        routes.append({'d':d,'amplitude':[float(amp.real),float(amp.imag)],'abs':float(abs(amp)),'phase':float(np.angle(amp)),
                       'spin_diagonal_amplitudes':[[float(z.real),float(z.imag)] for z in vals]})
    return {'nmax':nmax,'gauge_dim':G,'joint_dim':2*G,'nnz_W':int(W.nnz),'nnz_Mchi':int(Mchi.nnz),
            'Mchi_class_adjacency_residual':adj_res,'W_interior_isometry_fro_residual':winter,
            'kato':{'eigenvalue':[lam.real,lam.imag],'min_step_overlap':minov,'min_phase_gap':mingap,'eigen_residual':eigres},
            'routes':routes,'seconds':time.time()-t}

def main():
    cases=[]
    for n in (6,8,10,12,14):
        r=run(n);cases.append(r);print('nmax',n,'G',r['gauge_dim'],'routes',[(x['d'],x['abs']) for x in r['routes']],'sec',r['seconds'],flush=True)
    conv={}
    for d in (8,9,10):
        zs=[complex(*next(x for x in c['routes'] if x['d']==d)['amplitude']) for c in cases]
        conv[str(d)]={'last_change_abs':float(abs(zs[-1]-zs[-2])),'peak_to_peak_from_n10':float(max(abs(z-zs[-1]) for z in zs[2:]))}
    checks={'Mchi_character_action':bool(max(c['Mchi_class_adjacency_residual'] for c in cases)<1e-12),
            'W_interior_isometry':bool(max(c['W_interior_isometry_fro_residual'] for c in cases)<1e-10),
            'Kato_eigen':bool(max(c['kato']['eigen_residual'] for c in cases)<1e-10),
            'cutoff_route_stability':bool(max(conv[str(d)]['last_change_abs'] for d in (8,9,10))<1e-6)}
    obj={'status':'PASS_SU2_FULL_LINK_MATTER_ROUTE_BENCHMARK' if all(checks.values()) else 'CHECK_SU2_FULL_LINK_MATTER_ROUTE_BENCHMARK',
         'external_particle_data_used':False,
         'definition':'Single compact SU(2) link in full Peter-Weyl basis |jmn>, coupled to a two-component fundamental matter index by the unitary multiplication transporter D^(1/2)(U). Magnetic multiplication is K_B=32I-16 chi_1/2 and electric K_E=4C2=n(n+2). The neutral Kato branch is embedded exactly from the class-function subspace. A simplified route step applies the fundamental stream then the gauge electric+magnetic Floquet, repeated d=8,9,10 and divided by the neutral Kato eigenphase^d.',
         'scope':'Noncommuting one-link gauge+matter benchmark. It validates the full Peter-Weyl carrier semantics beyond the scalar-character rotor but is not the 24-link 3D PEPO contraction.',
         'cases':cases,'convergence':conv,'checks':checks}
    txt=json.dumps(obj,indent=2,sort_keys=True)+'\n';OUT.write_text(txt)
    print(json.dumps({'status':obj['status'],'frozen':cases[-1],'convergence':conv,'checks':checks,'sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2)[:12000])
if __name__=='__main__':main()
