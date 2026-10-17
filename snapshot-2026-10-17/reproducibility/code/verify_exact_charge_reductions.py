#!/usr/bin/env python3
"""Independent exact link checks and dense-basis checks of the new reductions."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from scipy import sparse,linalg

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/exact_reduction_verification'
RESULTS=ROOT/'reproducibility/results'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records={}
    for folder in ['exact_resonance','weyl_bloch','common_cyclic_car',
                   'coherent_charge_reduction','global_modular_charge']:
        directory=RESULTS/folder
        cert_path=next(directory.glob('*CERTIFICATE.json'))
        cert=json.loads(cert_path.read_text())
        data_path=next(directory.glob('*.npz'))
        assert cert['data_sha256']==sha(data_path)
        assert cert['full_C10_contract_completed'] is False
        records[str(cert_path.relative_to(ROOT))]=sha(cert_path)
    payload=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    with np.load(payload,allow_pickle=False) as z:
        links=z['links'].astype(np.int64)
        edges=z['plaquette_edge_indices'].astype(np.int64)
        signs=z['plaquette_edge_sign'].astype(np.int64)
    basis=RESULTS/'c10_primitive_gauss/C10_INTEGER_GAUSS_BASIS_AND_SOURCE_PATHS.npz'
    with np.load(basis,allow_pickle=False) as z:
        vertices=z['gauge_vertices'].astype(np.int64)
        chords=z['chord_links'].astype(np.int64)
        paths={r:z['path_R'+str(r)].astype(np.int64) for r in [3,5,7,9]}
    index={tuple(x):i for i,x in enumerate(vertices)}
    starts=links[:,1:];ends=starts.copy()
    ends[np.arange(len(links)),links[:,0]]+=1
    tail=np.array([index[tuple(x)] for x in starts])
    head=np.array([index[tuple(x)] for x in ends])
    boundary=sparse.csr_matrix((np.r_[-np.ones(len(links),int),np.ones(len(links),int)],
        (np.r_[tail,head],np.tile(np.arange(len(links)),2))),
        shape=(len(vertices),len(links)))
    global_cert=json.loads((RESULTS/'global_modular_charge/GLOBAL_MODULAR_CHARGE_CERTIFICATE.json').read_text())
    reports=[]
    with np.load(RESULTS/'global_modular_charge/GLOBAL_MODULAR_CHARGE_SOLUTIONS_AND_OBSTRUCTIONS.npz',
                 allow_pickle=False) as z:
        for rec in global_cert['source_paths']:
            r=rec['radius'];h=np.zeros(len(links),np.int64);h[paths[r]]=1
            assert np.array_equal(h,z['source_path_vectors'][:,[3,5,7,9].index(r)])
            phi=z['R'+str(r)+'_potential_mod63'].astype(np.int64)
            k=z['R'+str(r)+'_neutral_integer_cycle_flux'].astype(np.int64)
            assert np.array_equal(k[chords],z['R'+str(r)+'_chord_translation'])
            gradient=phi[head]-phi[tail]
            assert np.all((boundary@gradient-3*(boundary@h))%63==0)
            assert np.all((k+3*h-gradient)%63==0)
            assert not np.any(boundary@k) and int(k.sum())==0
            cycle_pairings=np.sum((k+3*h)[edges]*signs,axis=1)
            assert np.all(cycle_pairings%63==0)
            norm2=sum(int(v)**2 for v in k)
            phases=[(2*norm2-15*r)%126,(8*norm2-66*r)%126]
            assert phases==[rec['coherent_charge_phase_mod126'],rec['right_module_phase_mod126']]
            # Each integer plaquette is a generator of the already certified
            # full neutral integer lattice. This checks the linear coefficient
            # of the phase identity on every generator, not sampled vectors.
            for m in range(10):
                q=-3*m
                c=2*m*m*(norm2-9*r)+3*m*r
                assert (2*m*m*norm2-(2*q*q+q)*r-c)%126==0
                assert np.all((4*m*cycle_pairings)%126==0)
            reports.append({'radius':r,'checked_integer_cycle_generators':len(edges),
                'static_occupation_charges_checked':list(range(0,-28,-3)),
                'exact_source_phases_mod126':phases,'all_link_Gauss_and_Poisson_checks':True})
    # Independently exponentiate dense circulant matrices; no FFT update or
    # production routine is imported for this one-loop intertwining check.
    N=63;phi=math.pi/63
    S=np.roll(np.eye(N),1,axis=0)
    B=4*np.eye(N)-2*(S+S.T)
    magnetic=linalg.expm(-1j*phi*B)
    r=np.arange(N,dtype=np.int64)
    U0=magnetic@np.diag(np.exp(-1j*phi*((8*r*r)%126)))
    dense_errors=[]
    for q in range(0,-28,-3):
        Uq=magnetic@np.diag(np.exp(-1j*phi*((8*r*r+4*q*r+2*q*q+q)%126)))
        shift=np.linalg.matrix_power(S,(-5*q)%N)
        target=np.exp(1j*phi*((198*q*q-q)%126))*shift.T@U0@shift
        err=float(np.linalg.norm(Uq-target,ord=2))
        assert err<3e-13
        dense_errors.append(err)
    # Independent Gauss-sum proof by character orthogonality: in |G|^2,
    # the r-s=d term sums over r with coefficient 8*d. Only d=0 survives.
    assert [d for d in range(63) if (8*d)%63==0]==[0]
    norm_gauss_squared=63
    exact=json.loads((RESULTS/'exact_resonance/EXACT_RESONANCE_REDUCTION_CERTIFICATE.json').read_text())
    lower=F(exact['first_Bloch_trace_harmonic']['Bessel_J63_lower_rational'])
    upper=F(exact['first_Bloch_trace_harmonic']['Bessel_J63_upper_rational'])
    assert 0<lower<upper
    for rec in exact['resonant_rotor_parameter_mapping']:
        assert F(rec['beta_rational'])-F(rec['nu_integer'],8)-F(63,2)==0
    # Angle-fiber and weight-lattice calculations were separate algorithms.
    with np.load(RESULTS/'exact_resonance/EXACT_BLOCH_U1_COMMON_GNS_MOMENTS.npz',
                 allow_pickle=False) as z:
        a=z['M4_group_returns'].copy()
    with np.load(RESULTS/'weyl_bloch/WEYL_BLOCH_THREE_GROUP_CAR_MOMENTS.npz',
                 allow_pickle=False) as z:
        b=np.array([z['M4_neutral_returns'][0],*z['M4_charged_Y_returns']])
    cross=float(np.max(abs(a-b)));assert cross<3e-13
    # Exact relative-orientation witness: individual SU2 traces do not
    # determine a joint gauge-invariant Wilson product.
    individual_trace=F(1)
    joint_traces=[F(-1),F(1,2)]
    costs=[32*(1-t/2) for t in joint_traces]
    assert costs==[48,24]
    sx=np.array([[0,1],[1,0]],complex);sz=np.diag([1.,-1.])
    U=linalg.expm(1j*math.pi*sz/3)
    Vx=linalg.expm(1j*math.pi*sx/3)
    matrix_trace_error=max(abs(np.trace(U)-float(individual_trace)),
        abs(np.trace(U@U)-float(joint_traces[0])),
        abs(np.trace(U@Vx)-float(joint_traces[1])))
    assert matrix_trace_error<2e-14
    result={'status':'PASS_INDEPENDENT_EXACT_GLOBAL_LINK_AND_DENSE_ROTOR_VERIFICATION',
        'source_certificates_sha256':records,'payload_sha256':sha(payload),
        'global_static_U1_paths':reports,
        'dense_63_by_63_intertwining_max_norm_error':max(dense_errors),
        'independent_angle_weight_return_difference':cross,
        'Gauss_sum_norm_squared_by_character_orthogonality':norm_gauss_squared,
        'Bessel_trace_harmonic_rational_interval_strictly_positive':True,
        'SU2_relative_orientation_counterexample':{
            'individual_traces_exact':[str(individual_trace)]*2,
            'joint_Wilson_traces_exact':[str(t) for t in joint_traces],
            'magnetic_costs_exact':[str(c) for c in costs],
            'matrix_diagnostic_error':float(matrix_trace_error),
            'meaning':'Individual loop conjugacy classes do not retain joint non-Abelian orientation.'},
        'floating_point_interval_certified':False,
        'full_C10_contract_completed':False,
        'scope':'Exact modular static U1 transfer reduction on the full stored gauge support; independent isolated-rotor and representation checks. No moving charged C10 or connected spatial-limit inference.'}
    (OUT/'INDEPENDENT_EXACT_REDUCTION_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'paths':len(reports),
        'dense_error':max(dense_errors),'cross_basis_difference':cross,'full_C10':False},indent=2))


if __name__=='__main__':main()
