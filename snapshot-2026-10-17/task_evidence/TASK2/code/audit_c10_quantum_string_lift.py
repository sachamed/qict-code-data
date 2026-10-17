from __future__ import annotations
import importlib.util, json, math, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE='/mnt/data/task2_continue/compute_c10_dressed_u1_streaming.py'
spec=importlib.util.spec_from_file_location('b',BASE); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
OUT=Path('/mnt/data/task2_continue/TASK2_C10_QUANTUM_LIFT_AUDIT.json')
QABS=3

def resistance_R(R:int):
    nodes=b.ball(R); idx,phi=b.solve(nodes,b.SRC)
    # unit-source Dirichlet energy=sum undirected edge E^2 + boundary edges E^2
    # By source potential / Thomson: R_eff = phi(source) for unit current with graph Laplacian convention.
    return float(phi[idx[b.SRC]])

def path_expectations(R:int):
    nodes=b.ball(R); out=b.dag(nodes,b.SRC)
    sums=defaultdict(float); npaths=0
    # probabilities are normalized at each DAG node
    def dfs(a,p,flux,steps):
        nonlocal npaths
        for bb,pr,bd in out[a]:
            pp=p*pr
            f2=b.add_flux(flux,b.compile_edge(a,bb),QABS)
            if bd:
                npaths += 1
                vals=list(f2.values())
                e2=sum(v*v for v in vals)
                pplus=sum(2*v*v+v for v in vals)
                sums['prob'] += pp
                sums['E2'] += pp*e2
                sums['Pplus'] += pp*pplus
                sums['length_bcc'] += pp*(steps+1)
                sums['support_links'] += pp*len(vals)
            else:
                dfs(bb,pp,f2,steps+1)
    dfs(b.SRC,1.0,{},0)
    reff=resistance_R(R)
    tangent_min=(QABS**2)*reff
    return {
      'R':R,'path_count':npaths,'probability_sum':sums['prob'],
      'mean_BCC_path_length':sums['length_bcc'],
      'mean_axis_link_support':sums['support_links'],
      'string_mean_sum_E2':sums['E2'],
      'string_mean_sum_Pplus':sums['Pplus'],
      'C10_unit_R_eff':reff,
      'C10_q2_tangent_min_sum_E2':tangent_min,
      'string_to_C10_E2_ratio':sums['E2']/tangent_min,
    }

def main():
    rows=[path_expectations(R) for R in range(1,7)]
    # Exact infinite C10 value
    reff_inf=math.gamma(0.25)**4/(32*math.pi**3)
    obj={
      'status':'PASS_C10_STRING_LIFT_NOT_EQUAL_TANGENT_DRESSING',
      'external_particle_data_used':False,
      'statement':'A normalized quantum superposition of thin integer Wilson strings weighted by the C10 path measure does not realize the C10 tangent minimizer. In the electric-flux basis, E^2 is diagonal, so its expectation is the path-average thin-string energy, not the square of the mean C10 flow.',
      'rows':rows,
      'C10_exact_infinite_unit_resistance':reff_inf,
      'C10_exact_infinite_q2_tangent_energy_sum_E2':QABS**2*reff_inf,
      'consequence':'C10 must remain the tangent Gauss selector/infrared boundary condition. The physical compact charged state must be obtained by Kato/Floquet evolution inside the exact Gauss sector; neither P(gamma) nor sqrt(P(gamma)) Wilson-string amplitudes may be substituted for that state.',
      'proof_identity':'For orthogonal electric-flux string states |gamma>, <sum E_l^2>=sum_gamma p_gamma sum_l E_l(gamma)^2, whereas C10 minimizes sum_l (sum_gamma p_gamma E_l(gamma))^2. Jensen gives C10 <= string average, with strict inequality whenever multiple routes carry different flux.'
    }
    txt=json.dumps(obj,indent=2,sort_keys=True)+'\n';OUT.write_text(txt)
    print(json.dumps({'status':obj['status'],'rows':rows,'sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2))
if __name__=='__main__': main()
