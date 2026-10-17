from __future__ import annotations
import sys,json,argparse,time
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/task2_work/full/QICT_D21_COMPLETE_FIRST_PRINCIPLES_SUBMISSION')
sys.path.insert(0,str(ROOT/'code'/'final_closure'))
import verify_rank6_chiral_u1 as r6
from qict_route_first import electron_route_coefficients,I3,X,Z
D=[10,9,8]; WF=[I3,X,Z]; COEFF=electron_route_coefficients()

def run(N,steps=17,R=2):
    t=time.time(); vac,k=r6.kato_vacuum(R,N,steps=steps)
    KE,B,F=r6.dense_components(R,N,-3); Uq=r6.Uof(KE,B,F,1.0)
    gs=[]
    for d in D:
        Ud=np.linalg.matrix_power(Uq,d)
        gs.append(np.vdot(vac,Ud@vac))
    Bfam=sum(COEFF[a]*gs[a]*WF[a] for a in range(3))/3
    T=np.zeros((6,6),complex); T[:3,3:]=Bfam; T[3:,:3]=Bfam
    s=np.linalg.svd(T,compute_uv=False)
    leak=np.eye(6)-T.conj().T@T
    return {
      'N':N,'steps':steps,'R':R,'seconds':time.time()-t,
      'construction':'canonical temporal-semigroup charged compression: g_d=<Omega_Kato|U_q=-3^d|Omega_Kato>, d=(10,9,8), B=(c10 g10 I+c9 g9 X+c8 g8 Z)/3, T=[[0,B],[B,0]]',
      'route_lengths':D,
      'route_amplitudes':[[float(z.real),float(z.imag)] for z in gs],
      'route_abs':[float(abs(z)) for z in gs],
      'singular_values':s.tolist(),
      'K_UV':float(-np.mean(np.log(s))),
      'leakage_min_eig':float(np.linalg.eigvalsh((leak+leak.conj().T)/2).min()),
      'operator_norm':float(s.max()),
      'min_singular':float(s.min()),
      'kato':k,
      'T':[[[float(z.real),float(z.imag)] for z in row] for row in T]
    }

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--N',type=int,required=True); ap.add_argument('--steps',type=int,default=17); ap.add_argument('--out',required=True); a=ap.parse_args()
 out=run(a.N,a.steps); Path(a.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ['N','steps','seconds','route_abs','singular_values','K_UV','leakage_min_eig']},indent=2))
if __name__=='__main__':main()
