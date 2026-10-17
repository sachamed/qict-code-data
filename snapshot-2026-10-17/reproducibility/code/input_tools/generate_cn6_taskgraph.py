#!/usr/bin/env python3
"""Generate the certified multivolume task graph for QICT rank-six moments.

Each task is an immutable (L,n) target with the same physical definition
C_n^(6)(L)=V^dagger U_L^n V.  Tasks are emitted only inside the source-certified
no-wrap window.  No numerical moment is synthesized by this scheduler.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from certified_causal_cone_rank6 import no_wrap_depth,character_cutoffs,cone_counts

VOLUMES=(24,32,48,64,96,128)

def build():
    tasks=[]
    for L in VOLUMES:
        nmax=no_wrap_depth(L)
        cut=character_cutoffs(L,nmax)
        for n in range(9,nmax+1):
            tasks.append({
                'task_id':f'L{L}_n{n:03d}', 'L':L, 'n':n,
                'no_wrap_certified':True,
                'cone':cone_counts(n),
                'cutoffs':{
                    'U1_abs_k_max':cut['U1_abs_k_max'],
                    'SU2_twice_j_max':cut['SU2_twice_j_max'],
                    'SU3_max_p_q':cut['SU3_max_p_q']},
                'volume_character_tail_bound':cut['representation_tail_bound'],
                'required_inputs':[
                    f'data/physical_g2/L{L}/KATO_BUFFER_<b>.npz (at least four nested buffers)',
                    f'data/physical_g2/L{L}/BUFFER_CONTRACTION_CERTIFICATE.json or certified direct tail majorant',
                    f'data/physical_g2/L{L}/C10_SHELL_MAPS_<b>.npz',
                    f'data/physical_g2/L{L}/BOUNDARY_TO_PATH_<b>.npz',
                    'iota-resolved recoupled core compatible with the frozen action',
                    'six-column isometry in order (L1,L2,L3,R1,R2,R3)'],
                'acceptance':[
                    'Kato boundary density supplied by an explicit PSD Gram factorization or an independently certified PSD lower bound',
                    'certified uniform protecting-gap lower bound on the entire Riesz contour, including the between-node Lipschitz loss',
                    'Riesz/operator error bounded independently of the contour quadrature diagnostic',
                    'certified summable Cauchy-tail trace-distance bound to the infinite-buffer limit below state budget; pairwise closeness is diagnostic only',
                    'total Frobenius error below declared moment budget',
                    'C11 exact unitary-moment Gram positivity; any quoted numerical margin or rank additionally requires independent propagated error bounds'],
                'status':'CERTIFIED_LARGE_VOLUME_C10_KATO_EXECUTION_STAGE',
                'mass_extraction_stage':'CERTIFIED_LARGE_VOLUME_C10_KATO_EXECUTION_STAGE'})
    return tasks

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',type=Path,required=True); ap.add_argument('--csv',type=Path,required=True); a=ap.parse_args()
    tasks=build()
    out={'status':'PASS_CN6_CERTIFIED_TASKGRAPH','task_count':len(tasks),'volumes':list(VOLUMES),
         'definition':'C_n^(6)(L)=V^dagger U_L^n V','source_order':['L1','L2','L3','R1','R2','R3'],
         'physical_mass_promoted':False,
         'tasks':tasks,'mass_extraction_stage':'CERTIFIED_LARGE_VOLUME_C10_KATO_EXECUTION_STAGE'}
    a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    with a.csv.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['task_id','L','n','U1_kmax','SU2_2jmax','SU3_pqmax','character_tail_bound','status'])
        for t in tasks:
            c=t['cutoffs']; w.writerow([t['task_id'],t['L'],t['n'],c['U1_abs_k_max'],c['SU2_twice_j_max'],c['SU3_max_p_q'],f"{t['volume_character_tail_bound']:.16e}",t['status']])
    print(json.dumps({'status':out['status'],'task_count':len(tasks),'by_volume':{str(L):sum(t['L']==L for t in tasks) for L in VOLUMES}},indent=2))
if __name__=='__main__': main()
