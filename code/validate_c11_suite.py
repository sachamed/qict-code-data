#!/usr/bin/env python3
"""Integrated C11 validation on the distributed rank-six matrix-measure benchmark."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmark',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    cdir=Path(__file__).resolve().parent/'c11'
    bp=loadmod('c11bp',cdir/'c11_block_pencil.py')
    sc=loadmod('c11sc',cdir/'c11_support_complexity_certificate.py')
    fe=loadmod('c11fe',cdir/'exact_matrix_moment_flat_extension.py')
    z=np.load(a.benchmark,allow_pickle=False); C=np.asarray(z['moments'],complex)
    bpa=bp.analyze(C)
    p18=bp.pencil(C,18,svd_rtol=1e-11)
    sca=sc.analyze(a.benchmark,max_depth=16,rtol=1e-11)
    T=bp.block_toeplitz(C,16); ev=np.linalg.eigvalsh(T)
    exact=fe.evaluate()
    result={
      'classification':'C11_MATRIX_SPECTRAL_VALIDATION',
      'matrix_dimension':int(C.shape[1]),
      'moment_depth':int(len(C)-1),
      'C0_identity_fro_error':float(np.linalg.norm(C[0]-np.eye(C.shape[1]),'fro')),
      'toeplitz_depth':16,
      'toeplitz_lambda_min_numeric':float(ev[0]),
      'toeplitz_lambda_max_numeric':float(ev[-1]),
      'structural_positivity_theorem':'For C_k=V^*U^kV with U unitary, the block Toeplitz matrix is a Gram matrix and is positive semidefinite.',
      'block_pencil':{
        'rank_growth_depths':[int(x['K']) for x in bpa['pencils']],
        'rank_growth_sequence':[int(x['numerical_rank_diagnostic']) for x in bpa['pencils']],
        'spectral_depth':18,
        'spectral_numerical_rank':int(p18['numerical_rank_diagnostic']),
        'spectral_unit_circle_max_deviation':float(p18['unit_circle_max_deviation_diagnostic']),
        'interpretation':'At depth 18 the 108-dimensional pencil resolves six residue directions for each of the 18 unit-circle atoms.'
      },
      'support_complexity':{
        'serialized_atom_count':int(sca['serialized_finite_volume_spectrum']['unique_atom_count']),
        'serialized_total_residue_trace':float(sca['serialized_finite_volume_spectrum']['total_residue_trace']),
        'toeplitz_rank_at_depth_16':int(sca['numerical_final_toeplitz_rank']),
        'depth_scan_rank_sequence':[int(x['rank']) for x in sca['toeplitz_depth_scan']]
      },
      'exact_flat_extension':exact,
      'scope':'Deterministic validation of the C11 moment, Toeplitz, block-pencil, support-complexity, and exact flat-extension maps on the distributed positive matrix-valued benchmark.'
    }
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
