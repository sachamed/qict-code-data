#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,hashlib
from pathlib import Path
from itertools import product
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; DATA=ROOT/'data'; CERT.mkdir(exist_ok=True)
OUT=CERT/'edge3d_M9_M17_M17_certificate.json'; CSV=DATA/'edge3d_topology_and_state_growth_R1_R5.csv'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
DIRS=list(product((-1,1),repeat=3))
def ball(R):
    seen={(0,0,0):0}; fr=[(0,0,0)]
    for d in range(R):
      nx=[]
      for x in fr:
       for v in DIRS:
        y=tuple(x[i]+v[i] for i in range(3))
        if y not in seen: seen[y]=d+1; nx.append(y)
      fr=nx
    return set(seen)
def counts(R):
    V=ball(R); E=sum(1 for x in V for v in DIRS if tuple(x[i]+v[i] for i in range(3)) in V)//2; return len(V),E,E-len(V)+1
def main():
    rows=[]
    for R in range(1,6):
      V,E,b=counts(R); rows.append({'R':R,'vertices':V,'internal_BCC_edges':E,'cycle_rank_beta1':b,'log10_flux_states_using_only_m_minus1_0_plus1':b*math.log10(3),'log10_flux_states_if_all_U1_harmonics_minus9_to9_independent':b*math.log10(19),'dense_complex128_log10_bytes_all_U1_harmonics':math.log10(16)+b*math.log10(19)})
    with CSV.open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert [r['cycle_rank_beta1'] for r in rows]==[0,30,126,324,660]
    cert={'status':'DEFINED_OBSERVABLE__COMPACT_3D_CONVERGENCE_REQUIRED','requested_harmonic_cutoffs':{'U(1)':9,'SU(2)':17,'SU(3)':17},
      'bcc_counts':rows,'M_is_not_a_Hilbert_cutoff':True,'additional_physical_parameter_needed':False,
      'additional_numerical_convergence_axis_needed':'gauge-invariant tensor/representation compression increased until observables converge',
      'physical_edge_assignment':'assigned by convergence of the compact three-dimensional C8J+C9+C10 matter-gauge contraction',
      'available_edge_values':[],
      'physical_edge_condition':'local harmonic bounds and lower-dimensional diagnostics do not define the compact three-dimensional interacting edge',
      'excluded_proxies':['C7 free phase 0.174019637484...','C10 BCC resistance 0.174150491210...','1xR strip phases'],
      'electron_mass_observable':'defined by the converged physical edge together with the independently determined receiver clock',
      'input_sha256':{'code/verification/verify_edge3d_convergence.py':sha(__file__)}}
    OUT.write_text(json.dumps(cert,indent=2)+'\n'); print(json.dumps(cert,indent=2))
if __name__=='__main__': main()
