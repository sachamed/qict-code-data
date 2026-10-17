#!/usr/bin/env python3
from __future__ import annotations
import sys, math, json
from pathlib import Path
import numpy as np
from scipy.linalg import schur
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import compute_c8_static_charge_strip as strip
PHI=math.pi/63

def dft_matrix(N):
    j=np.arange(N); return np.exp(-2j*np.pi*np.outer(j,j)/N)/np.sqrt(N)

def dense_components(R,N,q):
    KE,_,_,_,_=strip.electric_diag(R,N,q); B=strip.theta_B_diag(R,N).reshape(-1)
    F=dft_matrix(N); FR=F
    for _ in range(R-1): FR=np.kron(FR,F)
    return KE,B,FR

def Uof(KE,B,F,s):
    DE=np.exp(-1j*s*PHI*KE); DB=np.exp(-1j*s*PHI*B)
    return F.conj().T @ (DB[:,None]*(F*DE[None,:]))

def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

def analyze(R,N,steps):
    vals,g,_=strip.ground_ref(R,N); g=g/np.linalg.norm(g)
    KE,B,F=dense_components(R,N,0); prev=g.copy(); track=[]; minov=1.; mingap=10.
    for s in np.linspace(0.02,1.0,steps):
        U=Uof(KE,B,F,float(s)); T,Q=schur(U,output='complex'); lam=np.diag(T)
        ovs=np.abs(Q.conj().T@prev)**2; j=int(np.argmax(ovs)); v=Q[:,j]
        al=np.vdot(prev,v); v*=np.exp(-1j*np.angle(al))
        phase=float(wrap(-np.angle(lam[j]))); allph=wrap(-np.angle(lam)); dif=np.abs(wrap(allph-phase)); dif[j]=99
        gap=float(np.min(dif)); ov=float(ovs[j]); minov=min(minov,ov); mingap=min(mingap,gap)
        track.append({'s':float(s),'overlap_prev':ov,'phase':phase,'nearest_quasiphase_gap':gap})
        prev=v
    return {'R':R,'N':N,'dim':N**R,'primitive_E0':float(vals[0]),'primitive_gap':float(vals[1]-vals[0]),
            'min_step_overlap':minov,'min_quasiphase_gap':mingap,'final_overlap_with_primitive':float(abs(np.vdot(g,prev))**2),'track':track}

def main():
    out=[]
    for R,N,steps in [(1,13,25),(1,17,25),(2,13,15),(2,17,15)]:
        out.append(analyze(R,N,steps))
    root=HERE.parent
    root.joinpath('results/kato_vacuum_reproduction.json').write_text(json.dumps(out,indent=2)+'\n')
    summary=[{k:x[k] for k in x if k!='track'} for x in out]
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
