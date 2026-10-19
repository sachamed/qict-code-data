#!/usr/bin/env python3
"""Independent QICT finite-moment residue and scale-invariant consistency tests.

Reads *user-provided* QICT C9 six-channel moments and 9x9 gravity selftest;
no inferred particle mass, continuum gamma_R, or QICT cosmological constant.
"""
from __future__ import annotations
import argparse, io, json, math, hashlib
from pathlib import Path
import numpy as np
from scipy.constants import G, c, hbar


def fejer_atom_upper(moments, theta=0.0):
    """Matrix F_N(theta)>=mu({e^{i theta}}), for PSD unitary spectral measure.

    C_k = integral z^k dmu; moments C_0,...,C_{N-1}.
    F_N = N^-2 [N*C0 + sum_{k=1}^{N-1} (N-k) (e^{-iktheta} Ck + h.c.)].
    """
    C = np.asarray(moments, dtype=np.complex128)
    N, d, e = C.shape
    if d != e or N < 2:
        raise ValueError('moments must have shape (N,d,d), N>=2')
    if not np.all(np.isfinite(C)):
        raise ValueError('moments must be finite')
    if np.linalg.norm(C[0]-C[0].conj().T)>1e-8:
        raise ValueError('C0 not Hermitian')
    F = N*C[0].copy()
    for k in range(1,N):
        A = np.exp(-1j*k*theta)*C[k]
        F += (N-k)*(A+A.conj().T)
    F /= N*N
    F = .5*(F+F.conj().T)
    ev = np.linalg.eigvalsh(F)
    return F, ev



def inverse_toeplitz_atom_upper(moments, theta=0.0):
    """Upper bound from the full matrix Toeplitz form, using linear solves.

    C[k] = integral z**k dmu and Toeplitz block (j,k) = C[j-k].
    Requires strict positive definiteness of the finite moment matrix.
    """
    C=np.asarray(moments, dtype=np.complex128)
    N,d,_=C.shape
    T=np.block([[C[j-k] if j>=k else C[k-j].conj().T
                 for k in range(N)] for j in range(N)])
    T=(T+T.conj().T)/2
    lam=np.linalg.eigvalsh(T)
    if lam[0] <= 0:
        raise ArithmeticError('moment Toeplitz matrix is not strictly positive')
    A=np.kron(np.exp(1j*theta*np.arange(N))[:,None],np.eye(d))
    X=np.linalg.solve(T,A)
    gram=(A.conj().T@X)
    gram=(gram+gram.conj().T)/2
    B=np.linalg.solve(gram,np.eye(d))
    B=(B+B.conj().T)/2
    resid=float(np.linalg.norm(T@X-A,ord='fro')/np.linalg.norm(A,ord='fro'))
    E=np.linalg.eigvalsh(B)
    return B,E,{'toeplitz_min_eigenvalue':float(lam[0]),
                 'toeplitz_condition_2':float(lam[-1]/lam[0]),
                 'solve_relative_residual':resid}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path(__file__).with_name('FINITE_C9_COMMON_STATE_AND_MOMENTS.npz'))
    ap.add_argument('--gravity-data',type=Path,default=Path(__file__).with_name('SPECTRAL_GRAVITY_SELFTEST_INPUT.npz'))
    ap.add_argument('--json',type=Path,default=Path(__file__).with_name('operator_closure_results.json'))
    ap.add_argument('--figure',type=Path,default=Path(__file__).resolve().parents[1]/'../01_Main_Manuscript/figures/spectral_atom_bounds.pdf')
    args=ap.parse_args()
    d=np.load(args.data,allow_pickle=False)
    out={'scope':'FINITE_QICT_C9_MOMENT_BOUND_AND_OBS_GRAVITATIONAL_TARGET','source_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(), 'regulators':[]}
    fig_rows=[]
    for r in range(1,5):
        C=d[f'R{r}_moments']
        F,ev=fejer_atom_upper(C,0)
        C0=C[0]
        toeplitz=np.block([[C[abs(j-k)] if j>=k else C[abs(j-k)].conj().T for k in range(5)] for j in range(5)])
        # Check Toeplitz positivity for convention using any one of C_{j-k} or C_{k-j}
        tmin=float(np.linalg.eigvalsh(.5*(toeplitz+toeplitz.conj().T))[0])
        if ev[0]<-1e-8 or tmin<-1e-8:
            raise ArithmeticError('PSD checks failed')
        B,eb,bdiag=inverse_toeplitz_atom_upper(C,0.0)
        # Loewner comparison: optimized polynomial bound cannot exceed Fejer.
        max_gap=float(np.linalg.eigvalsh(F-B)[0])
        if max_gap < -1e-9:
            raise ArithmeticError('optimized bound exceeds Fejer')
        row={'regulator':f'R{r}' ,'moment_count':int(C.shape[0]), 'channel_count':int(C.shape[1]),
             'neutral_carrier_dimension':int(d[f'R{r}_joint_neutral_eigenvalues'].size),
             'min_eigenvalue':float(ev[0]), 'max_eigenvalue':float(ev[-1]),
             'first_three_diagonal':[float(x) for x in F.diagonal().real[:3]],
             'last_three_diagonal':[float(x) for x in F.diagonal().real[3:]],
             'C0_identity_frobenius_error':float(np.linalg.norm(C0-np.eye(C0.shape[0]))),
             'toeplitz_N5_min_eig':tmin,
             'inverse_toeplitz_atom_min_eigenvalue':float(eb[0]),
             'inverse_toeplitz_atom_max_eigenvalue':float(eb[-1]),
             'fejer_to_optimized_max_ratio':float(ev[-1]/eb[-1]),
             'fejer_minus_optimized_min_eigenvalue':max_gap,
             **bdiag}
        out['regulators'].append(row); fig_rows.append((r,ev[0],ev[-1],eb[-1]))
    # Planck Collaboration (2020), Lambda l_P^2 = (2.846 +/- 0.076)e-122, flat-base-LCDM.
    Llp=2.846e-122; sig_Llp=0.076e-122
    out['observational_gravity_target']={
      'definition':'epsilon_0/gamma_R^2 = 8*pi*Lambda*hbar*G/c^3 = 8*pi*Lambda*ell_P^2',
      'value':8*math.pi*Llp, 'one_sigma_observational':8*math.pi*sig_Llp,
      'lambda_m_inverse2':Llp/(hbar*G/c**3),
      'source':'Planck Collaboration, A&A 641 A6 (2020), Eq. (16) in Planck units',
      'qict_prediction':False}
    # Freeze-in from one dark daughter, MB equilibrium parent and constant g_*(T)
    constant=135/(8*math.pi**3*1.66)
    M_Pl=1.220890e19 # GeV, convention H=1.66 sqrt(g*) T^2 / M_Pl
    M_A=1000.0; Gamma_A=1e-20; gs=gr=106.75
    y_direct=constant*Gamma_A*M_Pl/(gs*math.sqrt(gr)*M_A*M_A)
    checks=[]
    for E_star in (5.,100.,10000.):
        epsA=M_A/E_star; deltaA=Gamma_A/E_star
        gamma=(M_Pl/E_star)**2/(8*math.pi)
        y_invariant=constant*math.sqrt(8*math.pi*gamma)*deltaA/(gs*math.sqrt(gr)*epsA**2)
        checks.append({'E_star_GeV':E_star,'epsilon_A':epsA,'delta_A':deltaA,'gamma_R_matching_reference_G':gamma,'Y_invariant':y_invariant,'relative_error':(y_invariant-y_direct)/y_direct})
        if abs(y_invariant/y_direct-1)>2e-14:raise ArithmeticError('scale cancellation failed')
    out['scale_free_freeze_in']={'identity':'Y_inf = [135 g_A/(8*pi^3*1.66*g_s*sqrt(g_r))]*delta_A*sqrt(8*pi*gamma_R)/epsilon_A^2',
       'benchmark_inputs':'M_A=1000 GeV, Gamma_A=1e-20 GeV, g_A=1, g_s=g_r=106.75, reference M_Pl=1.220890e19 GeV',
       'Y_direct':y_direct,'coordinate_invariance_checks':checks,'qict_prediction':False}
    # selftest finite response: uses the method certified elsewhere. Read input to validate only one q.
    grav=np.load(args.gravity_data,allow_pickle=False)
    out['gravity_selftest']={'qell':float(grav['qell']),'N_cells':float(grav['N_cells']),
       'independent_nonzero_momenta_in_input':1,'continuum_gamma_R_extractable':False,
       'physical_vacuum_epsilon_0_extractable':False,
       'source_sha256':hashlib.sha256(args.gravity_data.read_bytes()).hexdigest()}
    try:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
      import plot_style
      R=np.array([x[0] for x in fig_rows]); lo=np.array([x[1] for x in fig_rows]); hi=np.array([x[2] for x in fig_rows]); opt=np.array([x[3] for x in fig_rows]);
      fig,ax=plt.subplots(figsize=(4.9,3.3));ax.plot(R,hi,marker='o',label='Fejer maximum'); ax.plot(R,opt,marker='s',label='inverse-Toeplitz maximum');
      ax.set_xticks(R,[f'R{i}' for i in R]);ax.set_ylabel('Finite-moment atom-weight upper bound');ax.set_xlabel('Available C9 regulator');ax.grid(alpha=.25);ax.legend(fontsize=9);fig.tight_layout();args.figure.parent.mkdir(exist_ok=True,parents=True);plot_style.save_figure(fig,args.figure);plt.close(fig)
    except ImportError:out['plot_warning']='matplotlib unavailable; figure not generated'
    args.json.parent.mkdir(exist_ok=True,parents=True)
    args.json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'regulators':out['regulators'], 'I_lambda':out['observational_gravity_target']['value'],'Y_direct':y_direct,'ok':True},indent=2))

if __name__=='__main__':main()
