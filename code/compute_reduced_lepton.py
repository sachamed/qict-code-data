#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import mpmath as mp
import numpy as np

def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def rotor_value(N: int, su2_multiplier: float):
    # Primitive local SU(2) Peter-Weyl rotor: KE|n> = n(n+2)|n>,
    # KB = A I - (A/2) M_chi, with A fixed by the declared gauge action.
    n=np.arange(N+1,dtype=float)
    KE=np.diag(n*(n+2.0))
    M=np.zeros((N+1,N+1),float)
    idx=np.arange(N); M[idx,idx+1]=1.0; M[idx+1,idx]=1.0
    A=float(su2_multiplier); H=KE+A*np.eye(N+1)-(A/2.0)*M
    eig,vec=np.linalg.eigh(H); v=vec[:,0]
    return float(eig[0]), float(0.5*(v@M@v))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parameters',required=True); ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); par_path=Path(a.parameters); par=json.loads(par_path.read_text()); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    mp.mp.dps=80
    Ns=[6,8,10,12,16,20,24,32,48,64,80]
    A=float(par['compact_gauge_action_multipliers']['SU2'])
    rows=[]
    for N in Ns:
        E,W=rotor_value(N,A); rows.append({'N':N,'ground_energy':E,'W_half':W})
    W=mp.mpf(str(rows[-1]['W_half']))
    R=mp.gamma(mp.mpf(1)/4)**4/(32*mp.pi**3)
    phi=2*mp.pi/int(par['receiver_central_order'])
    alpha_fac=R*mp.sqrt(W)*mp.cos(phi)
    rb=par['reduced_lepton_benchmark']
    cycle=int(rb['route_cycle_length'])
    route_lengths=[int(x) for x in rb['route_lengths']]
    defects=[cycle-2*d for d in route_lengths]
    bridge=int(rb['primitive_positive_bridge_multiplier'])
    fluxes=[bridge*x for x in defects]
    assign={k:int(v) for k,v in rb['channel_defect_assignment'].items()}
    if sorted(assign.values()) != sorted(fluxes):
        raise ValueError('Channel defect assignment must be a permutation of the bridged route defects.')
    def cost(E): return 2*E*E+E
    costs={k:cost(v) for k,v in assign.items()}
    gap_mu_e=costs['electron']-costs['muon']
    gap_tau_mu=costs['muon']-costs['tau']
    gap_tau_e=costs['electron']-costs['tau']
    trace_dim=int(rb['CAR_trace_dimension'])
    delta=-(mp.mpf(1)/trace_dim)*mp.log(1-alpha_fac**3)
    alpha=alpha_fac+delta
    r21=mp.e**(gap_mu_e*alpha); r32=mp.e**(gap_tau_mu*alpha); r31=mp.e**(gap_tau_e*alpha)
    Q=(1+r21+r31)/(1+mp.sqrt(r21)+mp.sqrt(r31))**2
    spreads=[(r-1)/(r+1) for r in (r21,r32,r31)]
    result={
      'calculation':'Reduced charged-lepton linked-cluster invariants',
      'parameters_sha256':sha256(par_path),'precision_decimal_digits':80,
      'rotor_truncation_N_final':Ns[-1],'rotor_ground_energy':mp.nstr(rows[-1]['ground_energy'],18),
      'W_half':mp.nstr(W,30),'W_half_last_step_abs_change':abs(rows[-1]['W_half']-rows[-2]['W_half']),
      'R_BCC':mp.nstr(R,40),'phi_star':mp.nstr(phi,40),'alpha_fac':mp.nstr(alpha_fac,40),
      'route_cycle_length':cycle,'route_lengths':route_lengths,'route_defects':defects,
      'primitive_positive_bridge_multiplier':bridge,'channel_defect_assignment':assign,'oriented_electric_costs':costs,
      'logarithmic_cost_gaps':{'mu_over_e':gap_mu_e,'tau_over_mu':gap_tau_mu,'tau_over_e':gap_tau_e},
      'CAR_trace_dimension':trace_dim,
      'delta_alpha_CAR':mp.nstr(delta,40),'alpha_reduced':mp.nstr(alpha,40),
      'mu_over_e_reduced':mp.nstr(r21,40),'tau_over_mu_reduced':mp.nstr(r32,40),'tau_over_e_reduced':mp.nstr(r31,40),
      'koide_reduced':mp.nstr(Q,40),
      'hierarchy_spread_lower_bounds':{'mu_over_e':mp.nstr(spreads[0],30),'tau_over_mu':mp.nstr(spreads[1],30),'tau_over_e':mp.nstr(spreads[2],30)},
      'definition':'Dimensionless invariant of the explicitly specified reduced receiver/BCC/SU(2)/CAR construction.'}
    with (out/'REDUCED_LEPTON_INVARIANTS.json').open('w') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
    with (out/'SU2_ROTOR_CONVERGENCE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['N','ground_energy','W_half']); w.writeheader(); w.writerows(rows)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
