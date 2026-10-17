#!/usr/bin/env python3
"""Common U1 x SU2 coupled-plaquette source/state GNS and resolvent bounds.

The left channels are weak doublets. Nine CAR component operators, not
six scalar fermion modes, underlie the six receiver columns.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/common_coupled_electroweak'
PHI=math.pi/63


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def component_car():
    creators=[]
    for a in range(9):
        rows=[];cols=[];vals=[]
        for n in range(512):
            if not (n>>a)&1:
                rows.append(n|(1<<a));cols.append(n)
                vals.append((-1)**((n&((1<<a)-1)).bit_count()))
        creators.append(sparse.csr_matrix((np.array(vals,dtype=np.int64),(rows,cols)),
                                         shape=(512,512)))
    identity=sparse.eye(512,dtype=np.int64,format='csr')
    for a in range(9):
        for b in range(9):
            anticom=creators[a].T@creators[b]+creators[b]@creators[a].T
            if a==b:anticom-=identity
            anticom.eliminate_zeros();assert anticom.nnz==0
            ac=creators[a]@creators[b]+creators[b]@creators[a]
            ac.eliminate_zeros();assert ac.nnz==0
    source=np.zeros((512,3,6),complex)
    for a in range(3):
        for alpha in range(2):source[1<<(2*a+alpha),alpha,a]=1/math.sqrt(2)
        source[1<<(6+a),2,3+a]=1.
    matrix=source.reshape(-1,6)
    assert np.linalg.norm(matrix.conj().T@matrix-np.eye(6))<2e-15
    return matrix


def negative_weak(labels,seed,data,indices,indptr,expected):
    M=sparse.csr_matrix((data,indices,indptr),shape=(len(seed),len(seed)))
    E=np.array([3*int(a)*(int(a)+2)+3*int(b)*(int(b)+2)+int(c)*(int(c)+2)
                for a,b,c,*rest in labels],np.int64)
    inverse=np.exp(1j*PHI*(E%126))
    v=seed.copy();ret=[]
    for n in range(13):
        ret.append(np.vdot(seed,v))
        v=inverse*(np.exp(1j*PHI*64)*
          expm_multiply(-1j*PHI*16*M,v,traceA=0))
    error=float(np.max(abs(np.array(ret)-expected.conj())))
    assert error<3e-12
    return np.array(ret),error


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    weakdir=ROOT/'reproducibility/results/charged_su2_theta'
    ydir=ROOT/'reproducibility/results/common_cyclic_car'
    weak_cert_path=weakdir/'CHARGED_SU2_THETA_CERTIFICATE.json'
    ycertpath=ydir/'COMMON_CYCLIC_CAR_KATO_CERTIFICATE.json'
    wc=json.loads(weak_cert_path.read_text());yc=json.loads(ycertpath.read_text())
    wdata=weakdir/'CHARGED_SU2_THETA_SOURCES_AND_RETURNS.npz'
    ydata=ydir/'COMMON_CYCLIC_CAR_KATO_MOMENTS.npz'
    assert sha(wdata)==wc['data_sha256'] and sha(ydata)==yc['data_sha256']
    with np.load(wdata,allow_pickle=False) as z:
        a0=z['R60_neutral_returns'];af=z['R60_charged_returns']
        negative=[];negative_errors=[]
        for name,expected in [('neutral',a0),('charged',af)]:
            prefix='R60_neutral_' if name=='neutral' else 'R60_'
            minus,err=negative_weak(z[prefix+'labels'],
                z[prefix+'source'],
                z['R60_'+name+'_magnetic_csr_data'],
                z['R60_'+name+'_magnetic_csr_indices'],
                z['R60_'+name+'_magnetic_csr_indptr'],expected)
            negative.append(minus);negative_errors.append(err)
        d0=len(z['R60_neutral_labels']);df=len(z['R60_labels'])
    with np.load(ydata,allow_pickle=False) as z:y=z['L504_returns']
    left=y[0].conj()*y[1]*a0.conj()*af
    right=y[0].conj()*y[2]*abs(a0)**2
    c=np.array([np.diag([left[n]]*3+[right[n]]*3) for n in range(13)])
    minus_left=y[0]*y[1].conj()*negative[0].conj()*negative[1]
    minus_right=y[0]*y[2].conj()*abs(negative[0])**2
    cminus=np.array([np.diag([minus_left[n]]*3+[minus_right[n]]*3) for n in range(13)])
    inverse_error=float(np.max(abs(cminus-c.conj().transpose(0,2,1))))
    assert inverse_error<3e-12
    T=np.block([[c[j-i] if j>=i else c[i-j].conj().T for j in range(13)]
                for i in range(13)])
    toeplitz=float(np.linalg.eigvalsh(T).min());assert toeplitz>-1e-11
    gram=component_car();C0=float(np.linalg.norm(c[0]-np.eye(6)))
    assert C0<3e-12
    epsilon=F(wc['relative_weak_moment_state_and_alias_bound_rational'])+\
            F(yc['cutoffs'][-1]['infinite_flux_state_and_alias_moment_bound_rational'])
    ng=json.loads((ROOT/'reproducibility/results/coupled_su2_spin_network/COUPLED_SU2_SPIN_NETWORK_CERTIFICATE.json').read_text())
    yg=json.loads((ROOT/'reproducibility/results/coupled_boundary/COUPLED_BOUNDARY_GNS_CERTIFICATE.json').read_text())
    energy=F(ng['cutoffs'][-1]['rayleigh_rational'])+F(yg['cutoffs'][-1]['ground']['rayleigh_rational'])
    gap=min(F(ng['cutoffs'][-1]['gap_lower_rational']),
            F(yc['uniform_homotopy_gap_lower_rational']))
    resolvents=[];Gs=[];Sigmas=[]
    for z in [2,3,4]:
        powers=np.array([float(F(1,z**(n+1))) for n in range(13)])
        g=np.einsum('n,nab->ab',powers,c)
        A=c[1];sigma=z*np.eye(6)-A-np.linalg.inv(g)
        tail=F(1,z**13*(z-1))
        delta=tail+epsilon*sum((F(1,z**(n+1)) for n in range(13)),F(0))
        assert delta<F(1,z+1)
        self_error=epsilon+F((z+1)**2)*delta/(1-(z+1)*delta)
        resolvents.append({'real_z':z,
            'resolvent_operator_error_bound_rational':str(delta),
            'resolvent_operator_error_bound':float(delta),
            'self_energy_operator_error_bound_rational':str(self_error),
            'self_energy_operator_error_bound':float(self_error),
            'inverse_bound_proof':'Re G(z)>=I/(z+1) from the normalized unitary spectral measure.'})
        Gs.append(g);Sigmas.append(sigma)
    npz=OUT/'COMMON_COUPLED_ELECTROWEAK_GNS_RESOLVENTS.npz'
    np.savez_compressed(npz,moments=c,negative_moments=cminus,
        CAR_vacuum_and_dual_source_columns=gram,
        weak_neutral_returns=a0,weak_charged_returns=af,
        U1_returns=y,real_resolvent_points=np.array([2,3,4]),
        compressed_resolvents=np.array(Gs),Feshbach_self_energies=np.array(Sigmas))
    result={'status':'COMMON_COUPLED_ELECTROWEAK_GNS_WITH_ANALYTIC_RETURN_AND_RESOLVENT_BOUNDS',
        'actual_plaquettes':[2925,2908],'shared_source_link':10377,
        'groups':['U1','SU2'],'finite_regulator':{'U1_cyclic_length':504,
            'weak_outer_twice_spin_cutoff':60,'neutral_weak_dimension':d0,
            'charged_weak_dimension':df,
            'neutral_gauge_tensor_dimension':504**2*d0,
            'left_charged_gauge_tensor_dimension':504**2*df,
            'largest_propagated_factor_vector':max(504**2,df)},
        'primitive_neutral_energy_rayleigh_rational':str(energy),
        'endpoint_neutral_primitive_gap_lower_rational':str(gap),
        'CAR_component_count':9,'CAR_component_algebra_exact':True,
        'CAR_checks_scope':'Canonical component CAR on Fock(C^9) and a separate vacuum-plus-dual one-source Gram. No all-occupation weak Gauss dressing is inferred.',
        'all_occupation_weak_dressed_CAR_on_finite_regulator_certified':False,
        'left_channels':'Three weak doublets, each retaining its two source/dual indices.',
        'right_channels':'Three weak singlet components.',
        'receiver_source_Gram_exact':'I6',
        'relative_GNS_type':'conjugate(U0) tensor Uf; pure neutral returns retained in each group.',
        'moments_depth':12,'C0_Frobenius_diagnostic':C0,
        'Toeplitz_minimum_diagnostic':toeplitz,
        'inverse_moment_diagnostic':inverse_error,
        'weak_inverse_return_diagnostics':negative_errors,
        'analytic_state_and_regulator_moment_bound_rational':str(epsilon),
        'analytic_state_and_regulator_moment_bound':float(epsilon),
        'resolvent_and_self_energy_bounds':resolvents,
        'source_certificate_sha256':{
            str(weak_cert_path.relative_to(ROOT)):sha(weak_cert_path),
            str(ycertpath.relative_to(ROOT)):sha(ycertpath)},
        'data_sha256':sha(npz),'floating_point_interval_certified':False,
        'SU3_color_factor_included':False,
        'moving_matter_and_chiral_bridge_included':False,
        'connected_spatial_limit_certified':False,
        'full_C10_contract_completed':False,
        'scope':'One authentic linked two-plaquette electroweak gauge component with CAR multiplet sources; no full neutral color spectator, BCC moving charged evolution, mass, flavor or graviton assignment.'}
    (OUT/'COMMON_COUPLED_ELECTROWEAK_GNS_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'energy':float(energy),
        'gap_lower':float(gap),'moment_bound':float(epsilon),
        'C0_error':C0,'Toeplitz_minimum':toeplitz,
        'inverse_moment_error':inverse_error,
        'resolvent_bounds':[{k:r[k] for k in ['real_z','resolvent_operator_error_bound','self_energy_operator_error_bound']} for r in resolvents]},indent=2))


if __name__=='__main__':main()
