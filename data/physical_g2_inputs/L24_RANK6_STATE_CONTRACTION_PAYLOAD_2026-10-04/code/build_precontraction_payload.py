#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, numpy as np

ACTION_SHA='c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7'
SOURCE_ORDER=np.array(['L1','L2','L3','R1','R2','R3'])
FAMILY_PAIRS=np.array([[0,3],[1,4],[2,5]],dtype=np.int8)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_json(p): return json.loads(Path(p).read_text())

def main():
    root=Path(__file__).resolve().parents[1]
    c=root/'components'
    action=c/'QICT_LATTICE_ACTION_FREEZE_V3.canonical.json'
    if sha(action)!=ACTION_SHA: raise SystemExit('ACTION_SHA_MISMATCH')

    geo=np.load(c/'N9_RELATIVE_CONE_IR.npz',allow_pickle=False)
    tan=np.load(c/'L24_N9_EXACT_GLOBAL_FIRST_KATO_TANGENT_REEXEC_2026-10-04.npz',allow_pickle=False)
    su2=np.load(c/'L24_N9_EXACT_SU2_SECOND_KRYLOV_CONNECTED_FROM_CLASSES.npz',allow_pickle=False)
    dag=np.load(c/'C10_R12_LAYERED_DAG.npz',allow_pickle=False)
    s10=np.load(c/'C10_SHELL_R10_PLANE_SWEEP_IR.npz',allow_pickle=False)
    s11=np.load(c/'C10_SHELL_R11_PLANE_SWEEP_IR.npz',allow_pickle=False)
    s12=np.load(c/'C10_SHELL_R12_PLANE_SWEEP_IR.npz',allow_pickle=False)

    ft=load_json(c/'L24_N9_EXACT_GLOBAL_FIRST_KATO_TANGENT_REEXEC_2026-10-04.json')
    su2j=load_json(c/'L24_N9_EXACT_SU2_SECOND_KRYLOV_CONNECTED_FROM_CLASSES.json')
    loc=load_json(c/'L24_N9_EXACT_CHARGED_LOCAL_LAYER.json')
    loc2=load_json(c/'L24_N9_CHARGED_LOCAL_BLOCKS_DETERMINISTIC.json')
    seed=load_json(c/'L24_N9_EXACT_SECOND_ORDER_KATO_SEED_REEXEC_2026-10-04.json')
    c10=load_json(c/'C10_R12_SHELL_GEOMETRY_CERTIFICATE.json')
    sweep=load_json(c/'C10_SHELL_PLANE_SWEEP_CERTIFICATE.json')

    gates={
      'action_hash': sha(action)==ACTION_SHA,
      'n9_geometry': int(geo['n'])==9 and len(geo['links'])==21136 and len(geo['plaquettes'])==18252,
      'first_kato_tangent': ft.get('status')=='PASS_L24_N9_EXACT_GLOBAL_FIRST_KATO_TANGENT',
      'su2_connected_decompressed': su2j.get('status')=='PASS_EXACT_CLASS_DECOMPRESSION_TO_CONNECTED_PAIR_PAYLOAD' and len(su2['pairs'])==261346,
      'charged_local_layer': loc.get('status')=='PASS_L24_EXACT_CHARGED_LOCAL_LAYER',
      'charged_local_blocks': loc2.get('status')=='PASS_L24_N9_CHARGED_LOCAL_BLOCKS_DETERMINISTIC',
      'second_order_seed': seed.get('status')=='PASS_L24_N9_EXACT_SECOND_ORDER_KATO_SEED',
      'c10_geometry': c10.get('status')=='PASS_EXACT_C10_R12_LAYERED_DAG_AND_SHELL_GEOMETRY',
      'c10_plane_sweep': sweep.get('status')=='PASS_C10_SHELL_CAUSAL_PLANE_SWEEP_IR',
    }
    if not all(gates.values()): raise SystemExit('PRECONTRACTION_GATE_FAILURE:'+json.dumps(gates,sort_keys=True))

    outnpz=root/'L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    np.savez_compressed(
      outnpz,
      source_order=SOURCE_ORDER,
      family_pairs=FAMILY_PAIRS,
      n=np.int64(9), L=np.int64(24),
      links=np.asarray(tan['links']),
      plaquettes=np.asarray(tan['plaquettes']),
      plaquette_edge_indices=np.asarray(tan['plaquette_edge_indices']),
      plaquette_edge_sign=np.asarray(tan['plaquette_edge_sign']),
      plaquette_intervals=np.asarray(tan['plaquette_intervals']),
      first_kato_species=np.asarray(tan['species']),
      first_kato_species_amplitudes=np.asarray(tan['species_amplitudes']),
      first_kato_start_offsets=np.asarray(tan['start_offsets']),
      first_kato_start_plaquette_ids=np.asarray(tan['start_plaquette_ids']),
      first_kato_end_offsets=np.asarray(tan['end_offsets']),
      first_kato_end_plaquette_ids=np.asarray(tan['end_plaquette_ids']),
      su2_connected_pairs=np.asarray(su2['pairs']),
      su2_connected_topology=np.asarray(su2['topology']),
      su2_connected_branch_count=np.asarray(su2['branch_count']),
      su2_connected_branch_code=np.asarray(su2['branch_code']),
      su2_connected_branch_amplitudes=np.asarray(su2['branch_amplitudes']),
      su2_pair_class_id=np.asarray(su2['pair_class_id']),
      su2_class_signature=np.asarray(su2['class_signature']),
      c10_nodes=np.asarray(dag['nodes']),
      c10_graph_distance=np.asarray(dag['graph_distance']),
      c10_potential=np.asarray(dag['potential']),
      c10_edge_src=np.asarray(dag['edge_src']),
      c10_edge_dst=np.asarray(dag['edge_dst']),
      c10_edge_is_boundary=np.asarray(dag['edge_is_boundary']),
      c10_edge_flow=np.asarray(dag['edge_flow']),
      c10_edge_probability=np.asarray(dag['edge_probability']),
      c10_edge_sigma=np.asarray(dag['edge_sigma']),
      c10_edge_outer_shell=np.asarray(dag['edge_outer_shell']),
      c10_splitstep_points=np.asarray(dag['splitstep_points']),
      shell10_links=np.asarray(s10['links']), shell10_plaquettes=np.asarray(s10['plaquettes']), shell10_pe=np.asarray(s10['plaquette_edge_indices']), shell10_intervals=np.asarray(s10['plaquette_intervals']),
      shell11_links=np.asarray(s11['links']), shell11_plaquettes=np.asarray(s11['plaquettes']), shell11_pe=np.asarray(s11['plaquette_edge_indices']), shell11_intervals=np.asarray(s11['plaquette_intervals']),
      shell12_links=np.asarray(s12['links']), shell12_plaquettes=np.asarray(s12['plaquettes']), shell12_pe=np.asarray(s12['plaquette_edge_indices']), shell12_intervals=np.asarray(s12['plaquette_intervals']),
    )
    cert={
      'status':'PASS_EXACT_L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD',
      'physical_promotable':False,
      'is_selected_lambda1_kato_state':False,
      'is_C9_6x6':False,
      'is_G2':False,
      'L':24,'n':9,'action_sha256':ACTION_SHA,
      'source_order':SOURCE_ORDER.tolist(),'family_pairs_zero_based':FAMILY_PAIRS.tolist(),
      'gates':gates,
      'dimensions':{
        'links':int(len(tan['links'])),'plaquettes':int(len(tan['plaquettes'])),
        'su2_connected_pairs':int(len(su2['pairs'])),'c10_nodes':int(len(dag['nodes'])),'c10_edges':int(len(dag['edge_src']))
      },
      'payload_npz':outnpz.name,'payload_npz_sha256':sha(outnpz),
      'what_this_payload_contains':[
        'authentic L24 n=9 covering-lattice geometry',
        'exact first Floquet-Kato tangent geometry/species/amplitudes',
        'losslessly decompressed exact SU2 second-Krylov connected-pair recouplings',
        'exact charged local matter/Gauss/chiral certificates (sidecar JSON)',
        'exact C10 R12 geometry and shell plane-sweep IR'
      ],
      'missing_before_physical_rank6_state_or_moment':[
        'complete matrix-free charged U_cone application with PiCAR=1 and six source columns propagated jointly',
        'lambda=1 selected Riesz/Kato density with independently certified whole-contour gap',
        'at least four nested KATO_BUFFER states plus certified tail-to-limit majorant',
        'volume-specific quantum C10 shell CP/Kraus maps (geometry alone is insufficient)',
        'boundary-density-to-recoupling-path insertion tensor',
        'volume-matched explicit six-column core source/sink embedding',
        'complete common-space transfer/matvec needed to evaluate C_n^(6)=V^dagger U^n V'
      ],
      'promotion_guard':'This file is an exact precontraction input bundle only. It must never be renamed or promoted as KATO_BUFFER, CN6_MOMENTS, C9^(6), G2, epsilon_f, Z_f or physical fermion masses.'
    }
    (root/'L24_N9_CHARGED_RANK6_PRECONTRACTION_CERTIFICATE.json').write_text(json.dumps(cert,indent=2,sort_keys=True)+'\n')
    print(json.dumps(cert,indent=2,sort_keys=True))

if __name__=='__main__': main()
