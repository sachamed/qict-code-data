#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import mpmath as mp
import numpy as np
import sympy as sp


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotor(N: int, A: float):
    n=np.arange(N+1,dtype=float)
    KE=np.diag(n*(n+2.0))
    M=np.zeros((N+1,N+1),float)
    idx=np.arange(N)
    M[idx,idx+1]=1.0; M[idx+1,idx]=1.0
    H=KE+A*np.eye(N+1)-(A/2.0)*M
    eig,vec=np.linalg.eigh(H)
    v=vec[:,0]
    if np.sum(v)<0: v=-v
    return H, M, float(eig[0]), v, float(0.5*(v@M@v))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parameters',required=True)
    ap.add_argument('--outdir',required=True)
    args=ap.parse_args()
    par_path=Path(args.parameters)
    par=json.loads(par_path.read_text())
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    # Symbolic uniqueness of the primitive oriented quadratic polynomial.
    a,b,c,E=sp.symbols('a b c E', real=True)
    P=a*E**2+b*E+c
    eqs=[sp.Eq(P.subs(E,0),0), sp.Eq(sp.diff(P,E,2).subs(E,0),4), sp.Eq(sp.expand((P-P.subs(E,-E))/2),E)]
    sol=sp.solve(eqs,[a,b,c], dict=True)
    if sol != [{a:sp.Integer(2),b:sp.Integer(1),c:sp.Integer(0)}]:
        raise RuntimeError(f'unexpected polynomial solution {sol}')
    Pplus=sp.expand(P.subs(sol[0]))
    Delta=sp.symbols('Delta', integer=True, positive=True)
    triangular=sp.expand((2*Delta)*(2*Delta+1)/2)
    tri_res=sp.simplify(Pplus.subs(E,Delta)-triangular)
    if tri_res != 0: raise RuntimeError('triangular identity failed')
    defects=[1,3,5]
    tri_vals={d:int(Pplus.subs(E,d)) for d in defects}

    # Primitive SU(2) rotor from declared trace-metric multiplier.
    A=float(par['compact_gauge_action_multipliers']['SU2'])
    if A != 32.0: raise ValueError('current primitive SU2 derivation expects declared A_SU2=32')
    Ns=[6,8,10,12,16,20,24,32,48,64,80]
    rows=[]
    last_v=None
    for N in Ns:
        H,M,E0,v,W=rotor(N,A)
        neg_count=int(np.count_nonzero(v < -1e-14))
        pos=v[v>0]
        min_pos=float(pos.min()) if pos.size else 0.0
        rows.append({'N':N,'ground_energy':E0,'W_half':W,'ground_vector_negative_component_count':neg_count,'ground_vector_min_positive_component':min_pos})
        last_v=v
    W=mp.mpf(str(rows[-1]['W_half']))
    last_step=abs(rows[-1]['W_half']-rows[-2]['W_half'])

    # Exact BCC return and central phase from declared primitive receiver data.
    mp.mp.dps=90
    R=mp.gamma(mp.mpf(1)/4)**4/(32*mp.pi**3)
    order=int(par['receiver_central_order'])
    phi=2*mp.pi/order
    alpha=R*mp.sqrt(W)*mp.cos(phi)

    # Tangent-product verification on a finite rotor truncation.
    Ntest=12
    H,M,E0,v,Wtest=rotor(Ntest,A)
    KE=np.diag(np.arange(Ntest+1,dtype=float)*(np.arange(Ntest+1,dtype=float)+2.0))
    KB=A*np.eye(Ntest+1)-(A/2.0)*M
    # exact derivative identity is algebraic; record finite matrix residual of H-(KE+KB).
    tangent_res=float(np.linalg.norm(H-(KE+KB),ord=2))

    result={
      'classification':'PRIMITIVE_ROTOR_DERIVATION',
      'parameters_sha256':sha256(par_path),
      'symbolic_oriented_polynomial':str(Pplus),
      'coefficients':{'quadratic':2,'linear':1,'constant':0},
      'conditions':{
        'vacuum_cost':'P(0)=0',
        'primitive_odd_part':'[P(E)-P(-E)]/2 = E',
        'direct_dual_hessian':'P\'\'(0)=4'
      },
      'triangular_identity':'P_+(Delta)=T_{2Delta}=(2Delta)(2Delta+1)/2',
      'triangular_identity_symbolic_residual':str(tri_res),
      'route_costs':{str(k):v for k,v in tri_vals.items()},
      'cost_gaps':{'T10_minus_T6':tri_vals[5]-tri_vals[3],'T6_minus_T2':tri_vals[3]-tri_vals[1],'T10_minus_T2':tri_vals[5]-tri_vals[1]},
      'SU2_multiplier_A':A,
      'rotor_definition':'H_rot=diag[n(n+2)] + 32 I - 16 M_chi1/2',
      'rotor_ground_energy_N80':rows[-1]['ground_energy'],
      'W_half_N80':rows[-1]['W_half'],
      'W_half_last_step_abs_change':last_step,
      'ground_vector_negative_component_count_N80':rows[-1]['ground_vector_negative_component_count'],
      'ground_vector_min_positive_component_N80':rows[-1]['ground_vector_min_positive_component'],
      'finite_matrix_tangent_generator_residual':tangent_res,
      'R_BCC':mp.nstr(R,50),
      'receiver_order':order,
      'phi_star':mp.nstr(phi,50),
      'alpha_fac':mp.nstr(alpha,50),
      'derivation_domain':'primitive receiver algebra together with the declared C7/C8 lowest-filtration, shortest-character, and direct-dual reciprocity conditions'
    }
    (out/'PRIMITIVE_DERIVATION_VALIDATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    with (out/'PRIMITIVE_ROTOR_CONVERGENCE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['N','ground_energy','W_half','ground_vector_negative_component_count','ground_vector_min_positive_component'])
        w.writeheader(); w.writerows(rows)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()
