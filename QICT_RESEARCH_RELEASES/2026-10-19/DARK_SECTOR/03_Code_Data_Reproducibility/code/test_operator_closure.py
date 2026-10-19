"""Deterministic mathematical and reproducibility checks for the new QICT analysis."""
import json, math, numpy as np, tempfile, subprocess, sys
from pathlib import Path
from compute_operator_closure import fejer_atom_upper, inverse_toeplitz_atom_upper

root=Path(__file__).resolve().parent
# A spectral measure with a 0.2 atom at phase 0 and 0.8 weight at phase pi.
N=3
C=np.array([[[.2+.8*(-1)**k]] for k in range(N)],dtype=complex)
F,e=fejer_atom_upper(C)
assert np.isclose(e[0],.2+.8/9,rtol=0,atol=1e-14)
assert e[0]>=.2
# Gap pi gives q=1/9 and an exact lower bound in this example.
q=1/9
assert abs((e[0]-q)/(1-q)-.2)<1e-14
# A measure with no atom at phase 0 obeys same upper bound.
C0=np.array([[[(-1)**k]] for k in range(4)],dtype=complex)
F0,e0=fejer_atom_upper(C0)
assert e0[0]>=-1e-14
# Independent scalar positive spectral measure: finite atom plus continuous circle.
# Haar measure gives vanishing nonzero moments, so the exact atom weight is known.
weight=.07
Cmix=np.array([[[1.0 if k==0 else weight]] for k in range(7)],dtype=complex)
Bmix,emix,_=inverse_toeplitz_atom_upper(Cmix)
Fmix,fmix=fejer_atom_upper(Cmix)
assert emix[0]>=weight-1e-12 and emix[0]<=fmix[0]+1e-12
# Error-robust bound via inflated moment matrix, using a stated hypothetical epsilon.
N=Cmix.shape[0]
Tmix=np.block([[Cmix[j-k] if j>=k else Cmix[k-j].conj().T for k in range(N)] for j in range(N)])
eps=1.e-8
Amix=np.ones((N,1))
Brob=float(1/(Amix.conj().T@np.linalg.solve(Tmix+eps*np.eye(N),Amix))[0,0].real)
assert Brob+1e-12>=emix[0] and Brob>=weight
# Direct four-regulator tests on input C9 matrices.
D=np.load(root/'FINITE_C9_COMMON_STATE_AND_MOMENTS.npz',allow_pickle=False)
vals=[]
for j in range(1,5):
    Fn,en=fejer_atom_upper(D[f'R{j}_moments'])
    assert len(en)==6 and en.min()>=-1e-12 and en.max()<.02
    B,eb,diag=inverse_toeplitz_atom_upper(D[f'R{j}_moments'])
    assert eb.min()>0 and eb.max() <= en.max()+1e-12
    assert np.linalg.eigvalsh(Fn-B).min()>-1e-10
    assert diag['solve_relative_residual']<1e-9
    vals.append(float(en.max()))
assert abs(vals[-1]-.016083559475960678)<5e-14
B4,ev4,_=inverse_toeplitz_atom_upper(D['R4_moments'])
assert abs(float(ev4[-1])-.0019562730145542767)<1e-12
# Output JSON consistency and Planck number.
R=json.loads((root/'operator_closure_results.json').read_text())
assert abs(R['observational_gravity_target']['value']/(8*math.pi)-2.846e-122)<1e-135
assert max(abs(c['relative_error']) for c in R['scale_free_freeze_in']['coordinate_invariance_checks'])<2e-14
assert len(R['regulators'])==4
for j,row in enumerate(R['regulators'],1):
    B,evals,_=inverse_toeplitz_atom_upper(D[f'R{j}_moments'])
    assert abs(row['inverse_toeplitz_atom_max_eigenvalue']-float(evals[-1]))<2e-12
assert R['gravity_selftest']['continuum_gamma_R_extractable'] is False
print('PASS: Fejer and inverse-Toeplitz atom bounds; 4 C9 matrices; clock cancellation; reference target; gravity scope')
