#!/usr/bin/env python3
"""Exact compact-group gauge/cell sublayer and Gauss-dressed lepton CAR sources.

The first authentic L24 plaquette fixes the orientations.  The three compact
group coefficients, the A7 state rule and the cell gate come from the frozen
QICT sources.  This computes that physical subsystem, NOT the three-dimensional
stream, route-first interaction, overlap completion or exterior C10 state.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reproducibility/results/nonabelian_common_car'
DATA = ROOT/'reproducibility/data/common_physical_chain'
ACTION = ROOT/'reproducibility/data/physical_g2_inputs/L24_RANK6_STATE_CONTRACTION_PAYLOAD_2026-10-04/components/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json'
PHI = math.pi/63
DEPTH = 12
ORDERS = {'U1': 14, 'SU2': 23, 'SU3': 25}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sqrt_upper(value: F) -> F:
    with localcontext() as ctx:
        ctx.prec = 90
        d = (Decimal(value.numerator)/Decimal(value.denominator)).sqrt()
        out = F(d)
        if out*out < value: out += F(1, 10**80)
        assert out*out >= value
        return out


def negative_inertia(matrix: np.ndarray, shift: F) -> int:
    """Exact unpivoted symmetric LDL inertia; all pivots are checked nonzero."""
    n = len(matrix)
    a = [[F(int(matrix[i, j]))-(shift if i == j else 0)
          for j in range(n)] for i in range(n)]
    count = 0
    for k in range(n):
        pivot = a[k][k]
        assert pivot != 0, ('zero LDL pivot', k)
        count += pivot < 0
        for i in range(k+1, n):
            if not a[i][k]: continue
            factor = a[i][k]/pivot
            for j in range(i, n):
                if a[j][k]:
                    a[j][i] -= factor*a[j][k]
                    a[i][j] = a[j][i]
    return int(count)


def su3_labels(radius):
    return [(p, q) for p in range(radius+1) for q in range(radius+1-p)]


def su3_fundamental(p, q):
    return [(a, b) for a, b in [(p+1, q), (p-1, q+1), (p, q-1)]
            if a >= 0 and b >= 0]


def su3_all_neighbors(p, q):
    return su3_fundamental(p, q)+[(b, a) for a, b in su3_fundamental(q, p)]


def neutral_generator(group, radius):
    if group == 'U1':
        labels = [(r,) for r in range(-radius, radius+1)]
        diag = [8*r*r+4 for r, in labels]
        coefficient = 2
        neighbors = lambda x: [(x[0]-1,), (x[0]+1,)]
    elif group == 'SU2':
        labels = [(s,) for s in range(radius+1)]
        diag = [4*s*(s+2)+32 for s, in labels]
        coefficient = 16
        neighbors = lambda x: [(s,) for s in (x[0]-1, x[0]+1) if s >= 0]
    elif group == 'SU3':
        labels = su3_labels(radius)
        diag = [4*(p*p+q*q+p*q+3*p+3*q)+36 for p, q in labels]
        coefficient = 6
        neighbors = lambda x: su3_all_neighbors(*x)
    else: raise ValueError(group)
    index = {x:i for i, x in enumerate(labels)}
    matrix = np.diag(diag).astype(np.int64)
    for i, label in enumerate(labels):
        for target in neighbors(label):
            if target in index: matrix[index[target], i] -= coefficient
    assert np.array_equal(matrix, matrix.T)
    return labels, matrix, neighbors, coefficient


def endpoint_spectral_counts():
    records = {}
    # Q H Q >= Q K_E Q and ||P H Q|| <= beta. The exact Schur lower
    # matrix A-(x+beta^2/(d-x))I therefore counts eigenvalues below x.
    for group, radius, x, beta, d in [('SU2', 12, 40, 16, 780),
                                     ('SU3', 8, 42, 36, 352)]:
        labels, a, _, _ = neutral_generator(group, radius)
        if group == 'SU3':
            assert d == min(4*(p*p+q*q+p*q+3*p+3*q)
                            for p, q in su3_labels(radius+1) if p+q == radius+1)
        shift = F(x)+F(beta*beta, d-x)
        count = negative_inertia(a, shift)
        assert count == 1
        records[group] = {'finite_count_dimension':len(labels),
                          'excluded_electric_energy_lower':d,
                          'P_H_Q_norm_upper':beta,
                          'tested_first_excited_energy_lower':x,
                          'Schur_scalar_shift_rational':str(shift),
                          'exact_negative_inertia':count,
                          'argument':'D>=d and ||B||<=beta imply S(x)>=A-(x+beta^2/(d-x))I. '
                                     'One negative LDL pivot implies at most one infinite eigenvalue below x.'}
    return records


def ground_certificate(group, radius, excited_lower):
    labels, a, neighbors, coefficient = neutral_generator(group, radius)
    if group in ('U1', 'SU2'):
        e, v = eigh_tridiagonal(np.diag(a).astype(float),
                               np.diag(a, 1).astype(float),
                               select='i', select_range=(0, 0))
    else: e, v = eigh(a.astype(float), subset_by_index=(0, 0))
    vector = v[:, 0]
    if vector[np.argmax(abs(vector))] < 0: vector = -vector
    vector /= np.linalg.norm(vector)
    witness = [F(format(float(x), '.17g')) for x in vector]
    norm = sum(x*x for x in witness)
    hv = [sum(F(int(a[i, j]))*witness[j] for j in range(len(labels)))
          for i in range(len(labels))]
    mu = sum(x*y for x, y in zip(witness, hv))/norm
    outside = {}
    labelset = set(labels)
    for label, value in zip(labels, witness):
        for target in neighbors(label):
            if target not in labelset:
                outside[target] = outside.get(target, F(0))-coefficient*value
    r2 = (sum((y-mu*x)**2 for x, y in zip(witness, hv))
          +sum(x*x for x in outside.values()))/norm
    residual = sqrt_upper(r2)
    gap = F(excited_lower)-mu
    assert gap > 0 and residual < gap
    td = residual/gap
    record = {'rayleigh_rational':str(mu), 'energy':float(mu),
              'infinite_residual_squared_rational':str(r2),
              'infinite_residual_upper_rational':str(residual),
              'infinite_residual_upper':float(residual),
              'first_excited_energy_lower':excited_lower,
              'gap_lower_rational':str(gap), 'gap_lower':float(gap),
              'ground_energy_interval':[float(mu-residual),float(mu)],
              'state_trace_distance_upper_rational':str(td),
              'state_trace_distance_upper':float(td),
              'state_vector_distance_upper':float(math.sqrt(2)*td),
              'omitted_neighbor_components_included':len(outside),
              'support_labels':labels,
              'rational_witness_components':[str(x) for x in witness]}
    return labels, vector, record


def su2_spinor_labels(max_s):
    return [(s, t) for s in range(max_s+1) for t in (s-1, s+1) if t >= 0]


def cg_numerators(s, t, alpha):
    # Squared CG coefficients times (2k+1), in a fixed pseudoreal external
    # basis. Reversing both external indices does not alter the Haar products.
    out = {}
    for m in range(-t, t+1, 2):
        value = (t+alpha*m+2)//2 if s == t+1 else (t-alpha*m)//2
        if value: out[m] = value
    return out


def spinor_haar_numerator(first, second, shifts):
    # Normalized class Haar kernel: h_m=delta_m0-(delta_m2+delta_m,-2)/2.
    # All arithmetic is integer before the single dimension square root.
    s, t = first; r, u = second
    out = 0
    for alpha in (1, -1):
        p = cg_numerators(s, t, alpha)
        q = cg_numerators(r, u, alpha)
        for m, value in q.items():
            for shift in shifts:
                out += value*(2*p.get(m+shift, 0)-p.get(m+shift+2, 0)-p.get(m+shift-2, 0))
    return out


def spinor_fundamental_matrix(max_s):
    labels = su2_spinor_labels(max_s)
    index = {x:i for i,x in enumerate(labels)}
    rows, cols, values = [], [], []
    symbolic = []
    for col, label in enumerate(labels):
        s, t = label
        for new_t in (t-1, t+1):
            if new_t < 0: continue
            for new_s in (new_t-1, new_t+1):
                other = (new_s, new_t)
                if other not in index: continue
                numerator = spinor_haar_numerator(other, label, (-1,1))
                if not numerator: continue
                radicand = (s+1)*(t+1)*(new_s+1)*(new_t+1)
                value = numerator/(2*math.sqrt(radicand))
                rows.append(index[other]);cols.append(col);values.append(value)
                symbolic.append([index[other], col, numerator, radicand])
    matrix = sparse.csr_matrix((values,(rows,cols)),shape=(len(labels),len(labels)))
    assert sparse.linalg.norm(matrix-matrix.T) < 1e-13
    # Exact Gram identity on the low basis, independent of magnetic entries.
    low = su2_spinor_labels(12)
    for i, first in enumerate(low):
        for j, second in enumerate(low):
            numerator = spinor_haar_numerator(first, second, (0,))
            expected = 2*(first[0]+1)*(first[1]+1) if i == j else 0
            assert numerator == expected
    return labels, matrix, np.asarray(symbolic,dtype=np.int64)


def su3_fusion_matrix(radius):
    labels = su3_labels(radius)
    index = {x:i for i,x in enumerate(labels)}
    rows, cols = [], []
    for col, label in enumerate(labels):
        for target in su3_all_neighbors(*label):
            if target in index: rows.append(index[target]);cols.append(col)
    matrix = sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(labels),len(labels)))
    assert (matrix-matrix.T).nnz == 0
    return labels, matrix


def propagate_returns(seed, magnetic_fusion, coefficient, constant, electric):
    vector = seed.astype(complex).copy()
    returns = []
    norm_error = 0.
    phase = np.exp(-1j*PHI*(np.asarray(electric,dtype=np.int64)%126))
    magnetic = (1j*PHI*coefficient)*magnetic_fusion
    constant_phase = np.exp(-1j*PHI*constant)
    for n in range(DEPTH+1):
        returns.append(np.vdot(seed, vector))
        norm_error = max(norm_error, float(abs(np.linalg.norm(vector)-1)))
        if n < DEPTH:
            vector = constant_phase*expm_multiply(magnetic, phase*vector, traceA=0)
    # Reverse order is checked on a separate orbit to verify adjoint moments.
    inverse = seed.astype(complex).copy()
    inverse_error = 0.
    for n in range(DEPTH+1):
        inverse_error = max(inverse_error,float(abs(np.vdot(seed,inverse)-np.conj(returns[n]))))
        if n < DEPTH:
            inverse = np.conj(phase*constant_phase)*expm_multiply(-magnetic,inverse,traceA=0)
    assert norm_error < 2e-11 and inverse_error < 2e-11
    return np.asarray(returns), {'orbit_norm_error':norm_error,'adjoint_return_error':inverse_error}


def embed_vector(labels, small_labels, vector):
    index = {x:i for i,x in enumerate(labels)}
    out = np.zeros(len(labels),complex)
    for label, value in zip(small_labels,vector):out[index[label]] = value
    return out


def su2_source_embedding(neutral_seed, max_s):
    labels = su2_spinor_labels(max_s)
    out = np.zeros(len(labels),complex)
    for i, (s,t) in enumerate(labels):
        out[i] = math.sqrt((t+1)/(2*(s+1)))*neutral_seed[s]
    assert abs(np.linalg.norm(out)-1)<2e-14
    return out


def relative_toeplitz(moments):
    t = np.block([[moments[j-i] if j >= i else moments[i-j].conj().T
                   for j in range(DEPTH+1)] for i in range(DEPTH+1)])
    return float(np.linalg.eigvalsh(t).min())


def polynomial_tail(x, order):
    # Exact rational majorant for exp(x), including the geometric series tail.
    stop = 24
    upper = sum(x**j/math.factorial(j) for j in range(stop+1))
    upper += (x**(stop+1)/math.factorial(stop+1))/(1-x/F(stop+2))
    return upper*x**(order+1)/math.factorial(order+1)


def source_checks():
    # Full CAR sign identities on every occupation state of the 18 modes:
    # 3*(two weak components + one right mode)*two matter/receiver copies.
    nmode = 18
    states = np.arange(1<<nmode,dtype=np.uint32)
    parity = np.zeros_like(states,dtype=np.uint8)
    for shift in range(nmode):parity ^= ((states>>shift)&1).astype(np.uint8)
    def lower_parity(a):
        out=np.zeros_like(states,dtype=np.int8)
        for b in range(a):out ^= ((states>>b)&1).astype(np.int8)
        return 1-2*out
    signs=[lower_parity(a) for a in range(nmode)]
    for a in range(nmode):
        for b in range(nmode):
            occupied_a=(states>>a)&1; occupied_b=(states>>b)&1
            if a == b:
                assert np.all(occupied_a+(1-occupied_a)==1)
            else:
                # Creation on b flips the sign of mode a exactly if b<a;
                # annihilation on a flips the sign of b exactly if a<b.
                first=signs[b]*signs[a]*(1-2*int(b<a))
                second=signs[a]*signs[b]*(1-2*int(a<b))
                assert np.all(first+second==0)
    # Exact U(1) Gauss coefficients: first actual edge carries q, matter q
    # at its start, compensating boundary -q at its end.
    gauss=[{'matter_charge':q,'electric_first_edge':q,'start_G':-q+q,'end_G':q-q}
           for q in (-3,-6)]
    assert all(row['start_G']==row['end_G']==0 for row in gauss)
    # Independent SU(2) gauge covariance of a Wilson-dressed singlet pair.
    rng=np.random.default_rng(126)
    def su2():
        x=rng.normal(size=4);x/=np.linalg.norm(x)
        return np.array([[x[0]+1j*x[3], x[2]+1j*x[1]],[-x[2]+1j*x[1],x[0]-1j*x[3]]])
    covariance=0.
    for _ in range(32):
        left,right,u=su2(),su2(),su2()
        pulled=left.conj().T@u@right
        transformed=left@pulled@right.conj().T
        covariance=max(covariance,float(np.linalg.norm(transformed-u)))
    assert covariance<3e-15
    return {'CAR_modes':nmode,'full_Fock_occupation_states_checked':1<<nmode,
            'all_mode_pairs_checked':nmode*nmode,'CAR_integer_sign_error':0,
            'U1_Gauss_constraints':gauss,'SU2_Wilson_pair_covariance_error':covariance,
            'source_representations':{'L1_L2_L3':{'qY':-3,'twice_j':1,'SU3':[0,0]},
                                      'R1_R2_R3':{'qY':-6,'twice_j':0,'SU3':[0,0]}},
            'reference_matter_state':'empty matter and receiver Fock vacuum',
            'source_Gram_exact':'I6 by CAR vacuum and unitary Wilson dressing',
            'chirality_scope':'Standard-Model representation labels; BCC overlap dispersion not computed here'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    action=json.loads(ACTION.read_text())
    assert action['phi_star']=='pi/63'
    assert action['electric_generator']=={'U1':'2 E_Y^2 + E_Y','SU2':'4 C2(E_2)','SU3':'3 C2(E_3)'}
    representations=json.loads((ROOT/'reproducibility/results/SECTOR_PREDICTIONS.json').read_text())
    charge_records=[v for v in representations.values() if isinstance(v,dict) and 'integer_hypercharges_qY' in v]
    assert len(charge_records)==1
    assert charge_records[0]['integer_hypercharges_qY']['L']==-3
    assert charge_records[0]['integer_hypercharges_qY']['e']==-6
    payload=DATA/'L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    with np.load(payload,allow_pickle=False) as z:
        orientations=z['plaquette_edge_sign'][0].tolist()
        edges=z['plaquette_edge_indices'][0].tolist()
    assert orientations==[1,1,-1,-1]
    spectral=endpoint_spectral_counts()
    car=source_checks()
    print('exact spectral inertia and gauge/CAR checks passed',flush=True)
    tail={g:polynomial_tail(x,ORDERS[g]) for g,x in [('U1',F(2,5)),('SU2',F(8,5)),('SU3',F(9,5))]}
    # After dropping constant phases, x bounds phi*||nonconstant magnetic||.
    single_bound=DEPTH*sum(tail.values())*math.prod(1+float(t) for t in tail.values())**DEPTH
    relative_bound=2*float(single_bound)
    outputs={}; records=[]; previous=None
    for level,(ry,k2,r3) in enumerate([(8,12,8),(12,16,10),(16,20,12),(20,24,14)]):
        ly,vy,cy=ground_certificate('U1',ry,8)
        l2,v2,c2=ground_certificate('SU2',k2,40)
        l3,v3,c3=ground_certificate('SU3',r3,42)
        margin_y=DEPTH*ORDERS['U1']+2+level*2
        flux=np.arange(-ry-margin_y,ry+margin_y+1)
        yseed=embed_vector([(int(r),) for r in flux],ly,vy)
        yfusion=sparse.diags([np.ones(len(flux)-1),np.ones(len(flux)-1)],[-1,1],format='csr')
        yreturns={}; diagnostics={}
        for q in (0,-3,-6):
            yreturns[q],diagnostics['Y'+str(q)]=propagate_returns(yseed,yfusion,2,4,8*flux*flux+4*q*flux+2*q*q+q)
        maxs=k2+DEPTH*ORDERS['SU2']+2+level*2
        s=np.arange(maxs+1)
        seed2=embed_vector([(int(x),) for x in s],l2,v2)
        f2=sparse.diags([np.ones(maxs),np.ones(maxs)],[-1,1],format='csr')
        ret2,diagnostics['SU2neutral']=propagate_returns(seed2,f2,16,32,4*s*(s+2))
        chargedlabels,chargedfusion,symbols=spinor_fundamental_matrix(maxs)
        chargedseed=su2_source_embedding(seed2,maxs)
        chargedcost=np.array([3*s*(s+2)+t*(t+2) for s,t in chargedlabels],dtype=np.int64)
        ret2L,diagnostics['SU2charged']=propagate_returns(chargedseed,chargedfusion,16,32,chargedcost)
        max3=r3+DEPTH*ORDERS['SU3']+2+level*2
        labels3,f3=su3_fusion_matrix(max3)
        seed3=embed_vector(labels3,l3,v3)
        cost3=np.array([4*(p*p+q*q+p*q+3*p+3*q) for p,q in labels3],dtype=np.int64)
        ret3,diagnostics['SU3neutral']=propagate_returns(seed3,f3,6,36,cost3)
        left=np.conj(yreturns[0])*yreturns[-3]*np.conj(ret2)*ret2L*abs(ret3)**2
        right=np.conj(yreturns[0])*yreturns[-6]*abs(ret2)**2*abs(ret3)**2
        moments=np.array([np.diag([left[n]]*3+[right[n]]*3) for n in range(DEPTH+1)])
        cellmoments=moments.copy();cellmoments[1::2]=0
        assert np.linalg.norm(moments[0]-np.eye(6))<1e-13
        change=None if previous is None else float(max(np.linalg.norm(a-b,2) for a,b in zip(moments,previous)))
        previous=moments
        vector_error=sum(c['state_vector_distance_upper'] for c in [cy,c2,c3])
        error=4*vector_error+relative_bound
        stationary=math.sqrt(max(0.,1-abs(yreturns[0][1]*ret2[1]*ret3[1])**2))
        record={'level':level,'ground_cutoffs':{'U1':ry,'SU2_twice_j':k2,'SU3_p_plus_q':r3},
                'common_evolution_dimensions':{'U1':len(flux),'SU2_neutral':len(s),
                                               'SU2_charged':len(chargedlabels),'SU3':len(labels3)},
                'ground_certificates':{'U1':cy,'SU2':c2,'SU3':c3},
                'primitive_ground_energy':cy['energy']+c2['energy']+c3['energy'],
                'primitive_ground_gap_lower':min(cy['gap_lower'],c2['gap_lower'],c3['gap_lower']),
                'primitive_density_Floquet_stationarity_distance_diagnostic':stationary,
                'moment_successive_cutoff_operator_norm_change':change,
                'analytic_state_and_flux_error_upper':error,
                'C0_identity_frobenius_error':float(np.linalg.norm(moments[0]-np.eye(6))),
                'Toeplitz_min_eigenvalue_diagnostic':relative_toeplitz(moments),
                'cell_Toeplitz_min_eigenvalue_diagnostic':relative_toeplitz(cellmoments),
                'floating_point_diagnostics':diagnostics}
        records.append(record)
        prefix='L'+str(level)+'_'
        outputs[prefix+'gauge_Cn6']=moments
        outputs[prefix+'gauge_cell_Cn6']=cellmoments
        outputs[prefix+'Y_flux_grid']=flux
        outputs[prefix+'Y_ground_factor']=yseed
        outputs[prefix+'SU2_ground_factor']=seed2
        outputs[prefix+'SU2_charged_source_factor']=chargedseed
        outputs[prefix+'SU2_spinor_labels']=np.array(chargedlabels,dtype=np.int32)
        outputs[prefix+'SU2_magnetic_symbolic_edges']=symbols
        outputs[prefix+'SU3_irrep_labels']=np.array(labels3,dtype=np.int32)
        outputs[prefix+'SU3_ground_factor']=seed3
        outputs[prefix+'neutral_group_returns']=np.array([yreturns[0],ret2,ret3])
        outputs[prefix+'charged_group_returns']=np.array([yreturns[-3],yreturns[-6],ret2L])
        print(json.dumps({'level':level,'E0':record['primitive_ground_energy'],'gap_lower':record['primitive_ground_gap_lower'],'moment_change':change,'analytic_error':error}),flush=True)
    path=OUT/'NONABELIAN_COMMON_CAR_KATO_MOMENTS.npz'
    np.savez_compressed(path,**outputs)
    result={'status':'EXACT_COMPACT_GAUGE_CELL_SUBSYSTEM__COMMON_A7_KATO_AND_GAUSS_DRESSED_CAR_SOURCES',
            'scope':'One authentic L24-oriented plaquette with all three compact groups, '
                    'the empty matter/receiver reference, external compensating boundary representations '
                    'and the exact zero/one-particle restriction of the frozen cell gate. '
                    'The BCC streams, route-first interaction and exterior C10 vacuum are not computed.',
            'action_sha256':sha(ACTION),'L24_payload_sha256':sha(payload),
            'actual_plaquette_edges':edges,'orientations':orientations,
            'state_selection':'A7 primitive Hamiltonian K_E+K_B, not Floquet finite-rank Riesz or C9-dephased state',
            'source_order':['L1','L2','L3','R1','R2','R3'],
            'CAR_and_Gauss_checks':car,'endpoint_gap_counts':spectral,
            'common_carrier':'H0 direct sum (three copies HL) direct sum (three copies HR), all on the same group cutoffs; sources are maps H0->Hf',
            'GNS_transfer':'conjugate(U0) tensor Uf on HS(H0,Hf)',
            'U_gauge':'exp(-i phi K_B) exp(-i phi K_E)',
            'U_gauge_cell':'exp(-i phi K_B) exp(-i phi K_E) U_cell',
            'cell_source_return_factor':'(1+(-1)^n)/2 in the zero/one-particle sector; the double-occupation phase is inactive there',
            'six_source_Gram_exact':'I6','family_structure':'I3 within each representation; no generation splitting in this sublayer',
            'analytic_polynomial_tail_bounds_rational':{g:str(t) for g,t in tail.items()},
            'analytic_relative_padding_error_upper':relative_bound,'cutoff_results':records,
            'full_charged_C10_contract_completed':False,'physical_fermion_poles_or_masses_derived':False,
            'total_floating_point_arithmetic_interval_certified':False,
            'numerical_policy':'Exact integer/Fraction arithmetic proves basis overlaps, CAR, inertia and rational state residuals. '
                               'Sparse exponential actions and reported moment changes are floating-point diagnostics with analytic truncation errors separately recorded.',
            'generated_npz_sha256':sha(path)}
    (OUT/'NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'certificate':str((OUT/'NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json').relative_to(ROOT)),
                      'final_energy':records[-1]['primitive_ground_energy'],
                      'final_gap_lower':records[-1]['primitive_ground_gap_lower'],
                      'full_C10':False},indent=2))


if __name__=='__main__':main()
