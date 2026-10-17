#!/usr/bin/env python3
"""Finite archived B0 operator diagnostic and canonical *mathematical* dilation.

Its dilation is not the physical QICT charged U_cone; no mass is inferred.
"""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
file=ROOT/'data/common_physical_chain/SU2_B0_ISOLATED_RIESZ_KATO.npz'
with np.load(file,allow_pickle=False) as z:
    T=np.asarray(z['U'],complex)
    gram=np.asarray(z['gram'],complex)
assert T.ndim==2 and T.shape[0]==T.shape[1]
assert gram.shape[0]==T.shape[0]
A,s,Bh=np.linalg.svd(T,full_matrices=True)
B=Bh.conj().T
assert np.max(s)<1+1e-12,'the array is not a contraction within tolerance'
D=np.diag(np.sqrt(np.maximum(0.0,1-np.minimum(s,1)**2)))
DT=B@D@B.conj().T
DTs=A@D@A.conj().T
W=np.block([[T,DTs],[DT,-T.conj().T]])
N=len(T)
uerr=float(np.linalg.norm(T.conj().T@T-np.eye(N),'fro'))
werr=float(np.linalg.norm(W.conj().T@W-np.eye(2*N),'fro'))
first=float(np.linalg.norm(W[:N,:N]-T,'fro'))
second=float(np.linalg.norm((W@W)[:N,:N]-T@T,'fro'))
assert uerr>1e-4
assert werr<1e-11
assert first<1e-12
assert second>1e-4
assert np.linalg.norm(gram)>0
out={
 'classification':'ARCHIVED_B0_FINITE_OPERATOR_DILATION_DIAGNOSTIC',
 'source':'data/common_physical_chain/SU2_B0_ISOLATED_RIESZ_KATO.npz',
 'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
 'stored_operator_shape':list(T.shape),
 'unitarity_defect_fro':uerr,
 'singular_value_min':float(s[-1]),'singular_value_max':float(s[0]),
 'canonical_julia_dilation_shape':list(W.shape),
 'dilation_unitarity_defect_fro':werr,
 'first_moment_compression_error_fro':first,
 'second_moment_compression_vs_T_squared_fro':second,
 'does_not_identify_physical_charged_unitary':True,
 'interpretation':'The archived B0 finite operator is contractive but not unitary. The displayed Julia extension is a mathematical completion only; even its compressed second moment differs from T squared. No charged G2, physical mass or physical gravitational result is inferred.',
 'numerical_backend':'numpy.linalg.svd; float64/complex128; BLAS result rounded',
 'pass':True
}
output=ROOT/'results/ARCHIVED_B0_DILATION_DIAGNOSTIC.json'
output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('PASS B0 finite-operator diagnostic', 'U defect',uerr,'Julia defect',werr,'second moment mismatch',second)
