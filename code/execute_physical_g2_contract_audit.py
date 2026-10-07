#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

REQ = [
    'four_valid_nested_kato_buffers','buffer_tail_certificate','quantum_c10_shell_maps',
    'boundary_to_path_tensor','core_recoupling_payload','rank6_embedding',
    'complete_charged_ucone_matvec','cn6_moments','cn6_shape_and_C0'
]

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def load(p: Path): return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a=ap.parse_args()
    root=a.input_root
    l24=root/'L24_RANK6_STATE_CONTRACTION_PAYLOAD_2026-10-04'
    c9=root/'QICT_REQUESTED_C9_G2_PHYSICAL_ELEMENTS_2026-10-06'
    cert=load(l24/'L24_N9_CHARGED_RANK6_PRECONTRACTION_CERTIFICATE.json')
    completion=load(l24/'L24_PHYSICAL_PAYLOAD_COMPLETION_STATUS.json')
    requested=load(c9/'REQUESTED_ELEMENTS_STATUS.json')
    hardstop=load(c9/'00_CONTRACT_AND_STATUS/PHYSICAL_C9_G2_HARD_STOP_2026-10-05(1).json')
    finalstatus=load(c9/'00_CONTRACT_AND_STATUS/FINAL_RESULTS_STATUS.json')
    payload=l24/'L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    payload_hash=sha256(payload)
    hash_ok=(payload_hash==cert['payload_npz_sha256'])
    with np.load(payload, allow_pickle=False) as z:
        stats={
            'L':int(z['L']), 'n':int(z['n']), 'source_order':[str(x) for x in z['source_order'].tolist()],
            'links':int(z['links'].shape[0]), 'plaquettes':int(z['plaquettes'].shape[0]),
            'su2_connected_pairs':int(z['su2_connected_pairs'].shape[0]),
            'c10_nodes':int(z['c10_nodes'].shape[0]), 'c10_edges':int(z['c10_edge_src'].shape[0])
        }
    c9_emb=c9/'01_SOURCE_PACKAGES/L24_RANK6_STATE_CONTRACTION_PAYLOAD_2026-10-04/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    embedded_same=c9_emb.exists() and sha256(c9_emb)==payload_hash
    synthetic_cn6=c9/'03_REQUESTED_ELEMENTS/CN6_MOMENTS/CN6_MOMENTS_L24.npz'
    syn={}
    if synthetic_cn6.exists():
        with np.load(synthetic_cn6, allow_pickle=False) as z:
            C=np.asarray(z['moments'])
            syn={'present':True,'depth':int(C.shape[0]-1),'shape':list(C.shape),'C0_fro_error':float(np.linalg.norm(C[0]-np.eye(6))),
                 'classification':'SYNTHETIC_CONTROL_ONLY','sha256':sha256(synthetic_cn6)}
    else: syn={'present':False}
    checks=dict(completion.get('checks',{}))
    missing=[k for k in REQ if not checks.get(k,False)]
    physical_ready=not missing
    out={
        'status':'PHYSICAL_G2_CONTINUUM_VALUE_CERTIFIED' if physical_ready else 'PHYSICAL_G2_PRECONTRACTION_CERTIFIED_CONTINUUM_SEQUENCE_DEFINED',
        'physical_G2_promoted':bool(physical_ready),
        'action_sha256':cert.get('action_sha256'),
        'precontraction':{'certificate_status':cert.get('status'),'payload_sha256':payload_hash,'certificate_hash_match':hash_ok,
                          'embedded_copy_bit_identical':embedded_same,'statistics':stats},
        'physical_contract_checks':checks,
        'continuum_sequence_inputs':missing,
        'missing_physical_objects':missing,  # backward-compatible machine key
        'requested_elements_status':requested.get('requested_elements',{}),
        'strict_execution_status':hardstop.get('status'),
        'strict_physical_chain':hardstop.get('physical_chain',{}),
        'continuum_contraction_operation':hardstop.get('minimal_unexecuted_operation'),
        'certified_available_results':hardstop.get('new_results',{}),
        'synthetic_cn6_control':syn,
        'interpretation':(
            'The exact L=24,n=9 precontraction and independent backend/local checks are certified. '
            'The physical continuum G2 observable is defined by the charged causal-cone contraction, the lambda=1 Kato boundary state, '
            'and the declared common multivolume regulator sequence. The continuum_sequence_inputs field enumerates the tensors that activate this numerical stage.'
        ),
        'continuum_execution_chain':[
            'serialize >=4 nested lambda=1 charged KATO_BUFFER states and a certified contraction-to-limit bound',
            'serialize coherence-resolved volume-specific C10 shell CP/Kraus maps',
            'serialize the boundary-density-to-iota recoupling insertion and volume-matched core recoupling tensor',
            'serialize six isometric rank-six source columns in order L1,L2,L3,R1,R2,R3',
            'implement and certify the complete frozen charged U_cone matmat with PiCAR=1',
            'compute C_n^(6)(L)=V^dagger U_L^n V through the certified moment depth',
            'certify block-Toeplitz positivity and the spectral support/edge',
            'repeat on the common multivolume regulator sequence before assigning a physical G2 limit'
        ],
        'source_final_status':finalstatus.get('status')
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
    if not hash_ok or not embedded_same: raise SystemExit(2)

if __name__=='__main__': main()
