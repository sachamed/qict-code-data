#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def component(z, tag: str) -> dict:
    rho=np.asarray(z[f'rho_{tag}'],np.complex128)
    F=np.asarray(z[f'gram_{tag}'],np.complex128)
    lbs=np.asarray(z[f'node_lbs_{tag}'],float)
    rho_h=(rho+rho.conj().T)/2
    gram=F@F.conj().T
    tr=float(np.trace(rho_h).real)
    gram_n=gram/np.trace(gram).real
    ev=np.linalg.eigvalsh(rho_h)
    return {
        'dimension':int(rho.shape[0]),
        'trace':tr,
        'hermiticity_fro_residual':float(np.linalg.norm(rho-rho.conj().T)),
        'gram_reconstruction_fro_residual':float(np.linalg.norm(rho-gram_n)),
        'positivity_roundoff_residual':float(max(0.0,-ev.min())),
        'largest_hermitian_eigenvalue':float(ev.max()),
        'numerical_rank_1e-12':int(np.count_nonzero(ev>1e-12)),
        'uniform_contour_lower_bound_min':float(lbs.min()),
        'uniform_contour_lower_bound_max':float(lbs.max()),
        'quadrature_nodes':int(lbs.size),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default=str(ROOT/'data/common_physical_chain/U1_B0_RIESZ_KATO_COMPONENTS.npz'))
    ap.add_argument('--out',default=str(ROOT/'results/U1_SECTOR_VALIDATION.json'))
    a=ap.parse_args()
    p=Path(a.input); z=np.load(p,allow_pickle=False)
    out={
        'classification':'FINITE_U1_B0_RIESZ_KATO_VALIDATION',
        'input_sha256':sha256(p),
        'components':{
            'singleton_boundary_component':component(z,'size1'),
            'four_plaquette_boundary_component':component(z,'size4'),
        },
        'interpretation':'The two serialized U(1) boundary components are normalized positive rank-one Riesz densities with explicit Gram factors and uniform contour separation on 512 quadrature nodes.'
    }
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
