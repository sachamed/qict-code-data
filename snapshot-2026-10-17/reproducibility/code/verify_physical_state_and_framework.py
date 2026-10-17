#!/usr/bin/env python3
"""Independent original-basis state traces and exact source normalization."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from scipy import sparse
from scipy.linalg import schur,eigh

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/physical_state_verification'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pp=ROOT/'reproducibility/results/physical_two_time/PHYSICAL_TWO_TIME_KERNEL_CERTIFICATE.json'
    pc=json.loads(pp.read_text())
    pd=pp.parent/'PHYSICAL_TWO_TIME_LEPTON_KERNELS.npz'
    assert hashlib.sha256(pd.read_bytes()).hexdigest()==pc['data_sha256']
    with np.load(pd,allow_pickle=False) as z:
        K=z['physical_kernel'];cell=z['physical_kernel_with_cell']
        O=sparse.csr_matrix((z['source_csr_data'],z['source_csr_indices'],z['source_csr_indptr']),
                           shape=tuple(z['source_csr_shape'])).tocsc()
    wd=ROOT/'reproducibility/results/charged_su2_theta/CHARGED_SU2_THETA_SOURCES_AND_RETURNS.npz'
    with np.load(wd,allow_pickle=False) as z:
        nl=z['R60_neutral_labels'];fl=z['R60_labels']
    seen=set()
    for j,(a,b,J) in enumerate(nl):
        norm=F(0)
        for pos in range(O.indptr[j],O.indptr[j+1]):
            i=int(O.indices[pos]);aa,bb,c,JJ,KK=map(int,fl[i])
            assert (aa,bb,JJ,KK)==(int(a),int(b),int(J),int(J))
            assert i not in seen;seen.add(i)
            squared=F(c+1,2*(int(J)+1));norm+=squared
            assert abs(O.data[pos]**2-float(squared))<1e-15
        assert norm==1
    block=K.transpose(0,2,1,3).reshape(42,42)
    assert np.max(abs(block-block.conj().T))<1e-12
    assert np.linalg.eigvalsh(block).min()>0.0001160
    for t in range(7):assert np.linalg.norm(K[t,t]-np.eye(6))<1e-12
    assert pc['per_entry_analytic_state_and_regulator_bound']<1.073e-13
    assert not pc['floating_point_interval_certified']
    cd=ROOT/'reproducibility/results/finite_C9_common_state/FINITE_C9_COMMON_STATE_AND_MOMENTS.npz'
    cp=cd.parent/'FINITE_C9_COMMON_STATE_CERTIFICATE.json';cc=json.loads(cp.read_text())
    assert hashlib.sha256(cd.read_bytes()).hexdigest()==cc['data_sha256']
    checks=[]
    with np.load(cd,allow_pickle=False) as z:
        for R in (1,2):
            UY=z[f'R{R}_neutral_U1'];UW=z[f'R{R}_neutral_SU2']
            UYf=[z[f'R{R}_left_U1'],z[f'R{R}_right_U1']]
            UWf=z[f'R{R}_charged_SU2'];O=z[f'R{R}_weak_source']
            seed=np.kron(z[f'R{R}_primitive_U1'],z[f'R{R}_primitive_SU2'])
            U0=np.kron(UY,UW)
            T,Q=schur(U0,output='complex');lam=np.diag(T)
            assert np.linalg.norm(T-np.diag(lam))<2e-12
            # Direct global Schur basis, not the producer's two-factor basis.
            v=Q.conj().T@seed;rho_eig=np.zeros_like(U0)
            used=set()
            for j in range(len(lam)):
                if j in used:continue
                ids=np.flatnonzero(abs(lam-lam[j])<1e-12)
                used.update(map(int,ids))
                rho_eig[np.ix_(ids,ids)]=np.outer(v[ids],v[ids].conj())
            rho=Q@rho_eig@Q.conj().T
            assert np.linalg.eigvalsh(rho).min()>-1e-12
            stationarity=float(np.linalg.norm(U0@rho@U0.conj().T-rho))
            expected=z[f'R{R}_moments'];errors=[]
            y0=np.eye(len(UY),dtype=complex);w0=np.eye(len(UW),dtype=complex)
            yf=[y0.copy(),y0.copy()];wf=np.eye(len(UWf),dtype=complex)
            for n in range(13):
                Aleft=np.kron(y0@yf[0],w0@O.T@wf@O)
                Aright=np.kron(y0@yf[1],np.eye(len(UW)))
                left=np.einsum('ij,ji',rho,Aleft);right=np.einsum('ij,ji',rho,Aright)
                errors.extend([abs(left-expected[n,0,0]),abs(right-expected[n,3,3])])
                y0=y0@UY.conj().T;w0=w0@UW.conj().T
                yf=[U@p for U,p in zip(UYf,yf)];wf=UWf@wf
            err=float(max(errors));assert err<2e-11 and stationarity<2e-11
            checks.append({'radius':R,'global_Schur_stationarity_diagnostic':stationarity,
                'all_13_physical_moment_trace_errors_diagnostic':err})
    # Entangled, nonstationary spectator: directly verify operator cancellation.
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex)
    H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
    psi=np.array([1,0,0,1],complex)/math.sqrt(2);rho=np.outer(psi,psi.conj())
    U0=np.kron(X,H);Uf=np.kron(Y,H);spectator_error=0.
    for t in range(5):
        for s in range(5):
            full=np.trace(rho@np.linalg.matrix_power(U0,-t)@
                          np.linalg.matrix_power(Uf,t-s)@np.linalg.matrix_power(U0,s))
            reduced=np.trace(.5*np.linalg.matrix_power(X,-t)@
                             np.linalg.matrix_power(Y,t-s)@np.linalg.matrix_power(X,s))
            spectator_error=max(spectator_error,float(abs(full-reduced)))
    assert spectator_error<2e-14
    result={'status':'PASS_INDEPENDENT_PHYSICAL_STATE_AND_FRAMEWORK_CHECKS',
        'exact_rational_source_column_normalizations_checked':len(nl),
        'two_time_Gram_and_equal_time_tests':'PASS',
        'direct_global_Schur_C9_checks':checks,
        'entangled_spectator_cancellation_test':{'synthetic':True,'error_diagnostic':spectator_error},
        'floating_point_interval_certified':False,
        'full_physical_closure':False,
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (pp,pd,cp,cd)}}
    (OUT/'INDEPENDENT_PHYSICAL_STATE_FRAMEWORK_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
