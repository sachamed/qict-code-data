#!/usr/bin/env python3
"""A7 then C9, without a finite-rank Floquet selector, on four common charts.

Finite electric/spin truncations only. Joint neutral quasienergy clusters
are retained coherently; independent group dephasing is not assumed.
Numerical degeneracy grouping is diagnostic, not an interval certificate.
"""
from pathlib import Path
import hashlib, json, math
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, schur, expm
from compute_coupled_su2_spin_network import labels as neutral_labels, matrix as neutral_matrix
from compute_charged_su2_theta import labels as charged_labels, matrix as charged_matrix
from compute_physical_two_time_kernel import source_matrix

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/finite_C9_common_state'
PHI=math.pi/63
DEPTH=12
TOL=1e-12


def schur_basis(U):
    T,Q=schur(U,output='complex')
    lam=np.diag(T).copy()
    err=float(np.linalg.norm(T-np.diag(lam)))
    assert err<1e-10
    return lam,Q,err


def joint_clusters(lam):
    order=np.argsort(np.angle(lam))
    groups=[]
    for idx in order:
        if not groups or abs(lam[idx]-lam[groups[-1][0]])>TOL:
            groups.append([int(idx)])
        else: groups[-1].append(int(idx))
    if len(groups)>1 and abs(lam[groups[0][0]]-lam[groups[-1][0]])<=TOL:
        groups[0]=groups[-1]+groups[0];groups.pop()
    return groups


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[];arrays={};old=None
    for R in (1,2,3,4):
        yl=[(a,b) for a in range(-R,R+1) for b in range(-R,R+1)]
        yi={x:i for i,x in enumerate(yl)}
        M=np.zeros((len(yl),len(yl)))
        for j,(a,b) in enumerate(yl):
            for row in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                if row in yi:M[yi[row],j]=1
        EY=np.array([8*a*a+8*b*b-4*a*b for a,b in yl],float)
        BY=8*np.eye(len(yl))-2*M
        eigY,vecY=eigh(np.diag(EY)+BY)
        seedY=vecY[:,0]
        UY0=expm(-1j*PHI*BY)@np.diag(np.exp(-1j*PHI*EY))
        UYf=[]
        for q in (-3,-6):
            Ef=EY+np.array([4*q*(a-b)+2*q*q+q for a,b in yl])
            UYf.append(expm(-1j*PHI*BY)@np.diag(np.exp(-1j*PHI*Ef)))
        nl=neutral_labels(R);_,M0,E0=neutral_matrix(nl)
        fl=charged_labels(R);Mf,Ef,herm,entrydiag=charged_matrix(fl)
        M0=M0.toarray();Mf=Mf.toarray()
        O=source_matrix(np.array(nl),np.array(fl)).toarray()
        B0=64*np.eye(len(nl))-16*M0
        Bf=64*np.eye(len(fl))-16*Mf
        eigW,vecW=eigh(np.diag(E0)+B0)
        seedW=vecW[:,0]
        UW0=expm(-1j*PHI*B0)@np.diag(np.exp(-1j*PHI*E0))
        UWf=expm(-1j*PHI*Bf)@np.diag(np.exp(-1j*PHI*Ef))
        ly,QY,sy=schur_basis(UY0);lw,QW,sw=schur_basis(UW0)
        lam=(ly[:,None]*lw[None,:]).ravel()
        c=np.kron(QY.conj().T@seedY,QW.conj().T@seedW)
        groups=joint_clusters(lam)
        group_id=np.empty(len(lam),np.int32)
        for j,ids in enumerate(groups):group_id[ids]=j
        width=max(float(max(abs(lam[ids]-lam[ids[0]]))) for ids in groups)
        comm_bound=sum(2*np.linalg.norm(c[ids])*
            np.linalg.norm((lam[ids]-lam[ids[0]])*c[ids]) for ids in groups)
        active=sum(float(np.vdot(c[ids],c[ids]).real)>1e-26 for ids in groups)
        trace=sum(float(np.vdot(c[ids],c[ids]).real) for ids in groups)
        pY=[np.eye(len(yl),dtype=complex) for q in range(2)]
        pW=np.eye(len(fl),dtype=complex)
        moments=[];BYn_all=[];BWn_all=[]
        for n in range(DEPTH+1):
            by=[QY.conj().T@p@QY for p in pY]
            bw=QW.conj().T@O.T@pW@O@QW
            answer=[]
            for sector in range(2):
                value=0j
                for ids in groups:
                    ids=np.array(ids)
                    iy,iw=ids//len(nl),ids%len(nl)
                    bs=by[sector][iy[:,None],iy[None,:]]
                    if sector==0:bs=bs*bw[iw[:,None],iw[None,:]]
                    else:bs=bs*(iw[:,None]==iw[None,:])*(lw[iw][:,None]**n)
                    value+=np.vdot(c[ids],bs@(c[ids]*lam[ids]**(-n)))
                answer.append(value)
            moments.append(np.diag([answer[0]]*3+[answer[1]]*3))
            if n in (1,2):BYn_all.append(by);BWn_all.append(bw)
            pY=[U@p for U,p in zip(UYf,pY)];pW=UWf@pW
        moments=np.array(moments)
        C0=float(np.linalg.norm(moments[0]-np.eye(6)))
        T=np.block([[moments[j-i] if j>=i else moments[i-j].conj().T
            for j in range(DEPTH+1)] for i in range(DEPTH+1)])
        toe=float(np.linalg.eigvalsh(T).min())
        assert C0<1e-10 and toe>-1e-10
        independent=None;comm=None
        if R<=2:
            # Reconstruct the density in the original product basis and
            # evaluate both sqrt(rho) and linear physical traces there.
            Q=np.kron(QY,QW)
            rhoeig=np.zeros((len(lam),len(lam)),complex)
            for ids in groups:rhoeig[np.ix_(ids,ids)]=np.outer(c[ids],c[ids].conj())
            rho=Q@rhoeig@Q.conj().T
            ee,vv=eigh(rho);sq=(vv*np.sqrt(np.maximum(ee,0)))@vv.conj().T
            U0=np.kron(UY0,UW0)
            comm=float(np.linalg.norm(U0@rho@U0.conj().T-rho))
            discrepancies=[]
            for n in (1,2):
                inverse=np.linalg.matrix_power(U0,-n)
                for sector in (0,1):
                    weakB=BWn_all[n-1] if sector==0 else np.diag(lw**n)
                    B=Q@np.kron(BYn_all[n-1][sector],weakB)@Q.conj().T
                    physical=np.trace(rho@inverse@B)
                    fixed=np.trace(sq@B@sq@inverse)
                    discrepancies.extend([abs(physical-fixed),
                        abs(physical-moments[n,3*sector,3*sector])])
            independent=float(max(discrepancies))
            assert independent<2e-10 and comm<2e-10
        rec={'radius':R,'neutral_dimension':len(yl)*len(nl),
            'left_charged_dimension':len(yl)*len(fl),
            'primitive_energy_diagnostic':float(eigY[0]+eigW[0]),
            'primitive_gap_diagnostic':float(min(eigY[1]-eigY[0],eigW[1]-eigW[0])),
            'joint_quasienergy_clusters':len(groups),'density_nonzero_cluster_rank_diagnostic':active,
            'joint_cluster_max_width_diagnostic':width,'density_trace_diagnostic':trace,
            'joint_cluster_stationarity_trace_norm_majorant_diagnostic':float(comm_bound),
            'source_isometry_diagnostic':float(np.linalg.norm(O.T@O-np.eye(len(nl)))),
            'Schur_off_diagonal_diagnostics':[sy,sw],
            'density_stationarity_dense_diagnostic':comm,
            'independent_physical_and_fixed_GNS_trace_error_diagnostic':independent,
            'C0_identity_diagnostic':C0,'Toeplitz_minimum_diagnostic':toe,
            'successive_regulator_moment_change_diagnostic':None if old is None else float(np.max(abs(moments-old)))}
        records.append(rec);old=moments
        arrays[f'R{R}_moments']=moments
        arrays[f'R{R}_joint_neutral_eigenvalues']=lam
        arrays[f'R{R}_primitive_coefficients']=c
        arrays[f'R{R}_joint_cluster_ids']=group_id
        arrays[f'R{R}_neutral_U1']=UY0
        arrays[f'R{R}_left_U1']=UYf[0];arrays[f'R{R}_right_U1']=UYf[1]
        arrays[f'R{R}_neutral_SU2']=UW0;arrays[f'R{R}_charged_SU2']=UWf
        arrays[f'R{R}_weak_source']=O
        arrays[f'R{R}_primitive_U1']=seedY;arrays[f'R{R}_primitive_SU2']=seedW
        print(json.dumps(rec),flush=True)
    data=OUT/'FINITE_C9_COMMON_STATE_AND_MOMENTS.npz'
    np.savez_compressed(data,**arrays)
    result={'status':'EXECUTED_FINITE_A7_C9_COMMON_STATE_WITH_SIX_STATIC_LEPTON_SOURCES',
        'actual_plaquettes':[2925,2908],'source_link':10377,
        'selector':'A7 primitive ground followed by joint C9 Cesaro dephasing under the same neutral frozen-order Floquet circuit.',
        'finite_rank_Floquet_Riesz_selector_required':False,
        'joint_group_dephasing_keeps_degenerate_coherences':True,
        'source_Gram_exact':'I6 from the source/dual isometry and CAR vacuum labels',
        'canonical_CAR_components':9,
        'all_occupation_weak_Gauss_CAR_certified':False,
        'neutral_color_cancels_in_static_physical_correlators':True,
        'regulator_convention':'Gauge-invariant electric/spin compression followed by exponentiation of the Hermitian magnetic compression; not Haar-invariant finite-dimensional multiplication.',
        'representation_cutoffs':records,'moment_depth':DEPTH,
        'eigenvalue_grouping_chord_tolerance':TOL,
        'floating_point_interval_certified':False,
        'finite_state_and_stationary_GNS_theorem_exact':True,
        'computed_cluster_projectors_interval_certified':False,
        'Haar_normal_C9_limit_certified':False,
        'spatial_thermodynamic_limit_certified':False,
        'full_moving_C10_completed':False,
        'scope':'Four representation cutoffs of the same linked two-face static component. This is not four nested spatial Kato buffers.',
        'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
    (OUT/'FINITE_C9_COMMON_STATE_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
