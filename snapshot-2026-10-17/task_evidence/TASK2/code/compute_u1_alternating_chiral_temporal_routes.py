from __future__ import annotations
import sys,json,time,argparse,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/task2_work/full/QICT_D21_COMPLETE_FIRST_PRINCIPLES_SUBMISSION')
sys.path.insert(0,str(ROOT/'code'/'final_closure'))
import verify_rank6_chiral_u1 as r6
from qict_route_first import electron_route_coefficients,I3,X,Z
D=(10,9,8); WF=(I3,X,Z); COEFF=electron_route_coefficients()

def run(N:int,R:int=2,steps:int=17):
    t=time.time(); vac,k=r6.kato_vacuum(R,N,steps=steps)
    Us={}
    for q in (-3,-6):
        KE,B,F=r6.dense_components(R,N,q); Us[q]=r6.Uof(KE,B,F,1.0)
    n=len(vac)
    # S acts on [L,R] gauge vector: after gauge layer, exact Clebsch swap.
    # S = [[0,U_R],[U_L,0]]
    def applyS(v):
        L,Rv=v
        return (Us[-6]@Rv, Us[-3]@L)
    G=[]
    for d in D:
        gd=np.zeros((2,2),complex)
        for cin in range(2):
            state=[np.zeros(n,complex),np.zeros(n,complex)]; state[cin]=vac.copy()
            for _ in range(d): state=list(applyS(state))
            for cout in range(2): gd[cout,cin]=np.vdot(vac,state[cout])
        G.append(gd)
    T=np.zeros((6,6),complex)
    for a in range(3): T += COEFF[a]*np.kron(G[a],WF[a])/3.0
    s=np.linalg.svd(T,compute_uv=False)
    leak=(np.eye(6)-T.conj().T@T); leak=(leak+leak.conj().T)/2
    return {
      'N':N,'R':R,'kato_steps':steps,'seconds':time.time()-t,'kato':k,
      'definition':'S=[[0,U_qR=-6],[U_qL=-3,0]] with gauge layer first and local Clebsch chirality swap after each macro-step; G_d=<Omega_Kato|S^d|Omega_Kato>; T=(1/3) sum_a c_a G_d[a] tensor W_a, d=(10,9,8).',
      'route_lengths':list(D),'route_chiral_matrices':[[[[float(z.real),float(z.imag)] for z in row] for row in g] for g in G],
      'singular_values':[float(x) for x in s],
      'K_UV_diagnostic':float(-np.mean(np.log(s))),
      'operator_norm':float(s.max()),'min_singular':float(s.min()),
      'leakage_min_eig':float(np.linalg.eigvalsh(leak).min()),
      'T':[[[float(z.real),float(z.imag)] for z in row] for row in T]
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--Ns',default='13,15,17,19,21');ap.add_argument('--R',type=int,default=2);ap.add_argument('--steps',type=int,default=17);ap.add_argument('--out',default='/mnt/data/task2_continue/TASK2_U1_ALTERNATING_CHIRAL_TEMPORAL_ROUTES.json');a=ap.parse_args()
    cases=[]
    for N in map(int,a.Ns.split(',')):
        r=run(N,a.R,a.steps); cases.append(r); print(json.dumps({k:r[k] for k in ('N','seconds','singular_values','K_UV_diagnostic','leakage_min_eig')},indent=2),flush=True)
    vals=np.array([c['K_UV_diagnostic'] for c in cases]); sv=np.array([c['singular_values'] for c in cases])
    rep={'status':'PASS_U1_ALTERNATING_CHIRAL_TEMPORAL_ROUTE_DIAGNOSTIC','external_particle_data_used':False,
         'scope':'Gauss-reduced compact U(1) strip temporal regression with neutral Kato state and correct alternating external chiral charges. It omits full 3D SU(2), SU(3), CAR/receiver environment and therefore is not the final Task-2 K_UV.',
         'cases':cases,'convergence':{'K_spread':float(np.ptp(vals)),'max_singular_spread':float(np.max(np.ptp(sv,axis=0)))},
         'frozen_regression':cases[-1]}
    txt=json.dumps(rep,indent=2,sort_keys=True)+'\n';Path(a.out).write_text(txt);print('sha256',hashlib.sha256(txt.encode()).hexdigest())
if __name__=='__main__':main()
