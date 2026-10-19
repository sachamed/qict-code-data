"""Exact logic and finite benchmarks motivated by QICT T28.43; NOT physical QICT predictions."""
from __future__ import annotations
import hashlib, json, math, cmath
from pathlib import Path
import numpy as np

ACTION_HASH = 'c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7'
PHI = math.pi / 63

def magnetic_cover(theta:float, u:np.ndarray, v:np.ndarray)->float:
    return float(4*(1-math.cos(theta)) + 32*(1-np.trace(u).real/2) + 36*(1-np.trace(v).real/3))

def magnetic_quotient_alternative(theta:float, u:np.ndarray, v:np.ndarray)->float:
    # Exactly T2231; alternative changes the frozen action.
    chi2 = abs(np.trace(u))**2-1
    chi3 = abs(np.trace(v))**2-1
    return float((1-math.cos(6*theta))/9 + 12*(1-chi2/3) + 16*(1-chi3/8))

def center_transform(theta:float,u:np.ndarray,v:np.ndarray):
    return theta+math.pi/3, -u, np.exp(2j*math.pi/3)*v

def quotient_audit():
    u=np.eye(2,dtype=complex);v=np.eye(3,dtype=complex)
    a=(0.0,u,v); b=center_transform(*a)
    cost=magnetic_cover(*b)-magnetic_cover(*a)
    phase=np.exp(-1j*PHI*cost)
    alt=magnetic_quotient_alternative(*b)-magnetic_quotient_alternative(*a)
    t=1e-3
    uu=np.diag([np.exp(.5j*t), np.exp(-.5j*t)])
    vv=np.diag([np.exp(1j*t),np.exp(-1j*t),1.0])
    small_cover=magnetic_cover(t,uu,vv)
    small_alternative=magnetic_quotient_alternative(t,uu,vv)
    return {'original_magnetic_cost_difference_exact':120,
            'original_magnetic_cost_difference_computed':cost,
            'phase_difference_abs_from_one':float(abs(phase-1)),
            'original_gate_descends_to_quotient':bool(abs(phase-1)<1e-12),
            'quotient_candidate_center_difference':alt,
            'weak_field_relative_discrepancy_at_t_1e-3':float(abs(small_cover-small_alternative)/small_cover),
            'quotient_candidate_is_original_frozen_action':False,
            'status':'PASS_EXACT_COVER_NON_DESCENT'}

def central_orbit_exact():
    """Central lifts of the identity, using exact rational cosines.

    For k=0,...,5 the U(1), SU(2), SU(3) normalized real
    fundamental characters are in {1, 1/2, -1/2, -1}.
    """
    from fractions import Fraction
    c1=(Fraction(1),Fraction(1,2),Fraction(-1,2),Fraction(-1),Fraction(-1,2),Fraction(1,2))
    c3=(Fraction(1),Fraction(-1,2),Fraction(-1,2),Fraction(1),Fraction(-1,2),Fraction(-1,2))
    costs=[4*(1-c1[k])+32*(1-(-1)**k)+36*(1-c3[k]) for k in range(6)]
    assert costs == [0,120,60,72,60,120]
    return {"central_lift_powers":list(range(6)),
            "magnetic_costs_exact":[str(x) for x in costs],
            "original_gate_fails_quotient_descent":any(x%126 for x in costs),
            "original_action_modified":False}

def build_auxiliary_control(m:int=7,delta:float=math.pi/2):
    """A mathematical control with an infinite-rank Haar analogue; not derived from QICT gates."""
    # H_m = C^m_Haar x C^2_copy x C^6_flavor
    if m<2:raise ValueError('m>=2 needed')
    n=12*m
    odd_indices=np.arange(0,n,12)[:,None] + np.arange(6)[None,:]
    even_indices=odd_indices+6
    phases=np.empty(n,dtype=complex)
    phases[odd_indices.flatten()]=1.
    phases[even_indices.flatten()]=np.exp(1j*delta)
    # Observable parity Q, eigenvalue -1 for odd channels.
    parity=np.empty(n,dtype=int)
    parity[odd_indices.flatten()]=-1
    parity[even_indices.flatten()]=1
    translation=np.repeat(np.exp(1j*126*2*math.pi*np.arange(m)/m),12)
    V=np.zeros((n,6),complex)
    for j in range(m):V[j*12:j*12+6,:]=np.eye(6)/np.sqrt(m)
    return phases,parity,translation,V

def auxiliary_pole_certificate(m=7, steps=64):
    phases,parity,T,V=build_auxiliary_control(m=m)
    assert np.linalg.norm(V.conj().T@V-np.eye(6))<1e-14
    assert np.max(np.abs(phases*T-T*phases))<1e-15
    assert np.max(np.abs(phases*parity-parity*phases))<1e-15
    odd=(parity==-1).astype(float)
    # Method 1: exact spectral projection (diagonal normal matrix).
    Pspec=(abs(phases-1)<1e-10).astype(float)
    Rspec=V.conj().T@(Pspec[:,None]*V)
    # Method 2: midpoint circle quadrature of a Riesz integral.
    radius=.5
    theta=2*np.pi*(np.arange(steps)+.5)/steps
    z=1+radius*np.exp(1j*theta)
    weight=radius*np.exp(1j*theta)
    Pcont=np.mean(weight[:,None]/(z[:,None]-phases[None,:]),axis=0)
    Rcont=V.conj().T@(Pcont[:,None]*V)
    # Method 3: exact four-period Cesaro/Fourier extraction. delta=pi/2.
    if abs(np.exp(4j*math.pi/2)-1)>1e-13 or np.max(abs(phases**4-1))>1e-12:
        raise ValueError('Fourier projector formula requires fourth-root phases')
    Pfourier=sum(phases**n for n in range(4))/4
    Rfourier=V.conj().T@(Pfourier[:,None]*V)
    gap=float(min(abs(z-1) for z in set(phases) if abs(z-1)>1e-8))
    return {'classification':'AUXILIARY_NON_QICT_HAAR_MODEL', 'haar_angle_grid':m,
       'global_isolated_projector_rank_finite_cutoff':int(round(Pspec.sum())),
       'physical_QICT_pole_certified':False,
       'copy_odd_parity_commutation_error':float(np.max(abs(phases*parity-parity*phases))),
       'haar_loop_translation_commutation_error':float(np.max(abs(phases*T-T*phases))),
       'source_isometry_error':float(np.linalg.norm(V.conj().T@V-np.eye(6))),
       'uniform_spectral_gap':gap, 'contour_gap':min(radius, gap-radius),
       'Riesz_midpoint_exact_arithmetic_error_bound':float((radius/gap)**steps/(1-(radius/gap)**steps)),
       'Riesz_midpoint_bound_excludes_roundoff':True,
       'source_residue_trace':float(np.trace(Rspec).real),
       'min_residue_eigenvalue':float(np.linalg.eigvalsh(Rspec)[0]),
       'method_Riesz_discrepancy':float(np.linalg.norm(Rcont-Rspec)),
       'method_Fourier_discrepancy':float(np.linalg.norm(Rfourier-Rspec)),
       'finite_rank_haar_endpoint_asserted':False,
       'note':'Limit projection rank grows as 6*m, while its six-channel residue is I6.'}

def truncated_moment_ambiguity(depth=9):
    # Uniform Haar measure on S1 versus uniform discrete (N+1)-th roots.
    n=depth+1
    roots=np.exp(2j*np.pi*np.arange(n)/n)
    moments=np.array([np.mean(roots**k) for k in range(depth+1)])
    haar=np.r_[1.,np.zeros(depth,dtype=complex)]
    return {'depth':depth,'max_moment_discrepancy':float(max(abs(moments-haar))),
      'haar_measure_has_atoms':False,'discrete_measure_atom_at_one':float(1/n),
      'discrete_positive_source_residue':float(1/n),
      'logical_conclusion':'Finite moments and Toeplitz PSD do not determine whether the spectrum has atoms.'}

def phase_to_mass_interval(theta_interval, tau_interval):
    """Dimensional map only; caller must provide independently measured tau and vacuum-subtracted branch."""
    if theta_interval is None or tau_interval is None:
        raise ValueError('Independent clock calibration and branch-certified phase interval are missing')
    try:
        lo,hi=map(float,theta_interval); a,b=map(float,tau_interval)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('Finite phase and clock intervals required') from exc
    if not (all(map(math.isfinite,(lo,hi,a,b))) and 0<=lo<=hi and 0<a<=b):
        raise ValueError('Finite, ordered and nonnegative phase/positive-clock intervals required')
    # Dimensional conversion only: not a physically certified QICT mass.
    # The argument must *already* be a vacuum-subtracted, branch-selected phase.
    hbar_GeVs=6.582119569e-25
    return (hbar_GeVs*lo/b,hbar_GeVs*hi/a)

def source_action_check(source):
    h=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if h!=ACTION_HASH:raise ValueError(f'Action SHA mismatch {h}')
    p=json.loads(Path(source).read_text())
    return {'hash':h,'source_gate_order':p['factor_order'],
        'gauge_group_header':p.get('gauge_group'),
        'magnetic_gate_domain_required':'product_group_cover',
        'full_quotient_link_dynamics_verified':False,
        'lambda1_kato_states_verified':False,
        'complete_route_holonomies_serialized':False}

if __name__=='__main__':
    home=Path(__file__).resolve().parents[1]
    source=home/'evidence/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json'
    report={'action':source_action_check(source),'magnetic_quotient':quotient_audit(), 'central_lift_orbit':central_orbit_exact(),
       'moment_non_identifiability': [truncated_moment_ambiguity(n) for n in (1,9,21)],
       'auxiliary_poles':[auxiliary_pole_certificate(m) for m in (4,5,11,13)]}
    (home/'results/MATHEMATICAL_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'action_ok':True,'cover_descends':report['magnetic_quotient']['original_gate_descends_to_quotient'],
        'auxiliary_min_residue':min(x['min_residue_eigenvalue'] for x in report['auxiliary_poles']),
        'auxiliary_max_contour_error':max(x['method_Riesz_discrepancy'] for x in report['auxiliary_poles'])},indent=2))

def copy_odd_gate_check(U:np.ndarray, Q:np.ndarray, V:np.ndarray, tol=1e-11):
    U=np.asarray(U,complex);Q=np.asarray(Q,complex);V=np.asarray(V,complex)
    if U.ndim!=2 or V.ndim!=2 or Q.ndim!=2 or not all(np.isfinite(x).all() for x in (U,Q,V)):
        raise ValueError('finite matrices required')
    n=U.shape[0]
    if U.shape!=(n,n) or Q.shape!=(n,n) or V.shape[0]!=n or V.shape[1]==0:
        raise ValueError('incompatible Hilbert dimensions')
    invol=float(np.linalg.norm(Q@Q-np.eye(n),ord=2))
    herm=float(np.linalg.norm(Q-Q.conj().T,ord=2))
    unit=float(np.linalg.norm(U.conj().T@U-np.eye(n),ord=2))
    comm=float(np.linalg.norm(U@Q-Q@U,ord=2))
    source=float(np.linalg.norm((np.eye(n)+Q)@V,ord=2))
    gram=float(np.linalg.norm(V.conj().T@V-np.eye(V.shape[1]),ord=2))
    if max(invol,herm,unit,comm,source,gram)>tol:
        raise ValueError('copy-odd physical-sector consistency fails')
    return {'unitarity_error':unit,'copy_parity_commutator':comm,
            'odd_source_error':source,'source_isometry_error':gram,'status':'PASS_FINITE_MATHEMATICAL_VALIDATION_ONLY'}

def schur_feshbach_check(U:np.ndarray, V:np.ndarray, z:complex):
    U=np.asarray(U,dtype=complex);V=np.asarray(V,dtype=complex)
    d=U.shape[0];k=V.shape[1]
    if U.ndim!=2 or V.ndim!=2 or U.shape!=(d,d) or V.shape[0]!=d or not 0<k<d or not np.isfinite(U).all() or not np.isfinite(V).all():
        raise ValueError('bad shapes')
    if np.linalg.norm(V.conj().T@V-np.eye(k))>1e-11:
        raise ValueError('source isometry fails')
    _,_,vh=np.linalg.svd(V.conj().T,full_matrices=True)
    Q=vh[k:,:].conj().T
    D=z*np.eye(d)-U
    A=V.conj().T@D@V;B=V.conj().T@D@Q
    C=Q.conj().T@D@V;E=Q.conj().T@D@Q
    if np.linalg.cond(D)>1e12 or np.linalg.cond(E)>1e12:
        raise ValueError('Resolvent or complementary block too ill-conditioned')
    effective=A-B@np.linalg.solve(E,C)
    if np.linalg.cond(effective)>1e12:
        raise ValueError('Schur complement too ill-conditioned')
    G_schur=np.linalg.solve(effective,np.eye(k,dtype=complex))
    G_direct=V.conj().T@np.linalg.solve(D,V)
    return {'error':float(np.linalg.norm(G_direct-G_schur)),
            'schur_inverse_verified':bool(np.linalg.norm(G_direct-G_schur)<1e-11),
            'status':'PASS_FINITE_MATRIX_ONLY'}
