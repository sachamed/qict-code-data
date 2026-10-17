#!/usr/bin/env python3
"""Extract K_UV from a contracted 6x6 compression matrix with an error certificate."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def kuv_from_T(T:np.ndarray, operator_error:float=0.0):
    if T.shape!=(6,6): raise ValueError('T must be 6x6')
    s=np.linalg.svd(T,compute_uv=False)
    if np.any(s<=0): raise ValueError('T is singular; log determinant is not finite')
    K=float(-np.mean(np.log(s)))  # = -1/12 log det(T^dagger T)
    smin=float(np.min(s))
    if operator_error<0: raise ValueError('operator_error must be nonnegative')
    if operator_error>=smin:
        bound=float('inf')
    elif operator_error==0:
        bound=0.0
    else:
        bound=float(np.mean(operator_error/(s-operator_error)))
    return {
      'K_UV':K,
      'singular_values':s.tolist(),
      'min_singular_value':smin,
      'input_operator_norm_error_bound':operator_error,
      'K_absolute_error_bound_from_Weyl_log_bound':bound,
      'formula':'K_UV=-(1/12)log det(T^dagger T)=-mean_i log(s_i) for rank 6'
    }


def parse_matrix(path:Path):
    if path.suffix=='.npy': return np.load(path)
    obj=json.loads(path.read_text())
    raw=obj.get('T',obj)
    A=np.zeros((6,6),complex)
    for i,row in enumerate(raw):
      for j,z in enumerate(row):
        if isinstance(z,list): A[i,j]=complex(z[0],z[1])
        else: A[i,j]=complex(z)
    return A


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('matrix',nargs='?'); ap.add_argument('--error',type=float,default=0.0); a=ap.parse_args()
    if a.matrix:
        out=kuv_from_T(parse_matrix(Path(a.matrix)),a.error)
    else:
        # Self-test only; not a physical QICT value.
        s=np.array([.9,.8,.7,.6,.5,.4]); T=np.diag(s)
        out=kuv_from_T(T,1e-8); out['status']='PASS_SELF_TEST_ONLY'; out['physical_result']=False
        assert abs(out['K_UV'] + np.mean(np.log(s)))<1e-14
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
