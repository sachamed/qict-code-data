#!/usr/bin/env python3
"""Independent rational controls for source and continuum estimates.

The spin, CAR, shift and two-level systems are finite comparison witnesses.
The regulator schedule is conditional on the estimates stated in the paper.
None supplies the full moving charged C10 update or its selected state.
"""
from pathlib import Path
from fractions import Fraction as F
import csv
import hashlib
import json
import platform
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproducibility/results/continuum_source_controls'


def identity(n):
    return np.array([[F(int(i == j)) for j in range(n)] for i in range(n)], dtype=object)


def zeros(rows, columns):
    return np.full((rows, columns), F(0), dtype=object)


def equal(a, b):
    return all(x == y for x, y in zip(a.flat, b.flat)) and a.shape == b.shape


def positive2(a):
    assert equal(a, a.T)
    return a[0, 0] >= 0 and a[1, 1] >= 0 and a[0, 0]*a[1, 1] >= a[0, 1]*a[1, 0]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power(a, n):
    if n < 0:
        return power(a.T, -n)
    result = identity(len(a))
    for _ in range(n):
        result = result @ a
    return result


def state_source_checks():
    pairs = [(F(3,5),F(4,5)), (F(5,13),F(12,13)),
             (F(8,17),F(15,17)), (F(7,25),F(24,25)),
             (F(20,29),F(21,29))]
    u0 = np.array([[F(3,5),-F(4,5)],[F(4,5),F(3,5)]], dtype=object)
    uf = np.diag([F(1),F(-1)]).astype(object)
    w = np.kron(u0, uf)
    assert equal(w.T @ w, identity(4))
    frames, moments = [], []
    for x, y in pairs:
        assert x*x + y*y == 1
        a = np.array([[x,F(0)],[F(0),x],[F(0),y],[F(0),F(0)]], dtype=object)
        g = np.diag([x*x,F(1)]).astype(object)
        assert equal(a.T @ a, g)
        v = a @ np.diag([1/x,F(1)]).astype(object)
        assert equal(v.T @ v, identity(2))
        frames.append(v)
        c = [v.T @ power(w,n) @ v for n in range(13)]
        assert equal(c[0], identity(2))
        for n in range(13):
            assert equal(v.T @ power(w,-n) @ v,c[n].T)
        # A Toeplitz block is the Gram of unitary source orbits.
        orbit = [power(w,n) @ v for n in range(5)]
        for i in range(5):
            for j in range(5):
                assert equal(orbit[i].T @ orbit[j], v.T @ power(w,j-i) @ v)
        moments.append(c)
    checks = 0
    for i in range(len(pairs)):
        for j in range(i):
            x,y = pairs[i]; xp,yp = pairs[j]
            delta = 2*abs(x*x-xp*xp)
            dv2 = (x-xp)**2 + (y-yp)**2
            # Gram and source square-root bounds, all checked as rational squares.
            assert (x*x-xp*xp)**2 <= 16*delta
            assert 2*(x-xp)**2+(y-yp)**2 <= 2*delta
            g = min(x*x,xp*xp)
            assert dv2 <= 2*delta*(g+2)**2/g**3
            for n in range(13):
                dc = moments[i][n]-moments[j][n]
                assert positive2(4*dv2*identity(2)-dc.T @ dc)
                checks += 1
    # A nearby state continues a positive source Gram floor.
    t = F(1,10000)
    cosine,sine = (1-t*t)/(1+t*t),2*t/(1+t*t)
    x,y = pairs[0]
    xp,yp = cosine*x-sine*y,sine*x+cosine*y
    assert xp*xp+yp*yp == 1
    delta = 2*abs(x*x-xp*xp)
    floor = x*x
    assert 16*delta <= floor**2/4
    assert min(xp*xp,F(1)) >= floor/2
    return {'classification':'FINITE_RATIONAL_GNS_COMPARISONS',
            'density_pairs':len(pairs), 'moment_orders':list(range(13)),
            'moment_operator_inequalities_checked':checks,
            'normalization_and_adjoint_symmetry':'VERIFIED',
            'toeplitz_positivity':'EXACT_ORBIT_GRAM_FACTORIZATION',
            'reference_Gram_floor':str(floor),
            'continued_Gram_floor_bound':str(floor/2),
            'nearby_density_trace_distance_norm':str(delta)}


def car_checks():
    creation = []
    for a in range(6):
        c = np.zeros((64,64), dtype=np.int64)
        for state in range(64):
            if not state & (1 << a):
                c[state | (1 << a),state] = (-1)**((state & ((1 << a)-1)).bit_count())
        creation.append(c)
    eye = np.eye(64,dtype=np.int64)
    checks = 0
    for a in range(6):
        for b in range(6):
            assert np.array_equal(creation[a] @ creation[b] + creation[b] @ creation[a],0*eye)
            assert np.array_equal(creation[a].T @ creation[b] + creation[b] @ creation[a].T,
                                  int(a==b)*eye)
            checks += 2
    gram_records = []
    for eta in [F(1),F(1,4),F(1,16),F(1,64),F(1,256),F(0)]:
        g = zeros(6,6)
        for a in range(6):
            for b in range(6):
                prod = creation[a].T @ creation[b]
                g[a,b] = eta*int(prod[0,0])+(1-eta)*int(prod[63,63])
        assert equal(g,eta*identity(6))
        gram_records.append({'vacuum_weight':str(eta),'Gram_floor':str(eta),
                             'Gram_rank':6 if eta else 0})
    return {'classification':'SIX_MODE_FINITE_FOCK_COMPARISON',
            'Fock_dimension':64,'CAR_identities_checked':checks,
            'Gram_matrices':gram_records}


def boundary_checks():
    vertices = [(0,0,0)]+[(x,y,z) for x in [-1,1] for y in [-1,1] for z in [-1,1]]
    configurations = [[1 if state & (1 << x) else -1 for x in range(9)] for state in range(512)]
    records = []
    for numerator in range(17):
        s = F(numerator,16)
        energies = []
        for spins in configurations:
            bulk = sum((1-spins[0]*spins[x])//2 for x in range(1,9))
            boundary = sum((1-(1-2*s)*spins[x])/2 for x in range(1,9))
            energies.append(bulk+boundary)
        ordered = sorted(energies)
        rank = ordered.count(ordered[0])
        gap1 = ordered[1]-ordered[0]
        if numerator in [0,16]:
            assert rank==1 and gap1==2
            ground_index = energies.index(ordered[0])
            ground = configurations[ground_index]
            assert len(set(ground))==1
            probabilities = [int(i==ground_index) for i in range(512)]
            expectations = [sum(p*z[i] for p,z in zip(probabilities,configurations))
                            for i in range(9)]
            for i in range(9):
                for j in range(9):
                    if i!=j:
                        pair = sum(p*z[i]*z[j] for p,z in zip(probabilities,configurations))
                        assert pair-expectations[i]*expectations[j]==0
            assert expectations[0]==(1 if numerator==0 else -1)
        if numerator==8:
            assert rank==2 and gap1==0
        records.append({'interpolation':str(s),'ground_energy':str(ordered[0]),
                        'ground_multiplicity':rank,'rank_one_gap':str(gap1)})
    return {'classification':'PINNED_SPIN_BOUNDARY_COMPARISON',
            'graph':'BCC_NINE_VERTEX_STAR','vertices':vertices,
            'configurations_enumerated':512,'boundary_vertices':8,
            'endpoint_gaps':['2','2'],'bulk_Z_expectation_difference':'2',
            'disjoint_connected_correlations':'0',
            'linear_boundary_interpolation':records}


def moment_tail_checks():
    records = []
    for k in [9,13,21,29,45,61]:
        m = k+1
        for sign in [1,-1]:
            # Integer source orbit including the first signed return.
            orbit = []
            for n in range(m+1):
                v = [0]*m
                v[n % m] = sign**(n//m)
                orbit.append(v)
            assert [v[0] for v in orbit[:m]] == [1]+[0]*k
            assert orbit[m][0]==sign
            for i in range(m+1):
                for j in range(m+1):
                    inner = sum(x*y for x,y in zip(orbit[i],orbit[j]))
                    difference = abs(j-i)
                    expected = sign**(difference//m) if difference % m==0 else 0
                    assert inner==expected
        tail = F(1,2**k)
        separation = F(1,2**m)/(1-F(1,2**(2*m)))
        assert 0<separation<=tail
        records.append({'K':k,'unobserved_tail_fraction':str(tail),
                        'unobserved_tail_decimal':format(float(tail),'.16g'),
                        'rank6_carrier_dimension':6*m,
                        'signed_shift_metric_fraction':str(separation)})
    return records


def holder_check():
    source = np.zeros((24,6),dtype=np.int64)
    w = np.zeros((24,24),dtype=np.int64)
    for a in range(6):
        source[4*a,a],source[4*a+3,a] = 8,1
        for i in range(4):
            w[4*a+(3-i),4*a+i] = 1
    assert np.array_equal(source.T @ source,65*np.eye(6,dtype=np.int64))
    assert np.array_equal(source.T @ w @ source,16*np.eye(6,dtype=np.int64))
    squared_frobenius = F(6*16**2,65**2)
    squared_linear = F(12**2,65**2)
    assert squared_frobenius>squared_linear
    return {'classification':'SIX_ORTHOGONAL_SOURCE_COMPARISON',
            'epsilon':str(F(1,65)),'moment_difference_scalar':str(F(16,65)),
            'Frobenius_difference_squared':str(squared_frobenius),
            'squared_12D_bound':str(squared_linear)}


def ceil_rational(x):
    return -(-x.numerator//x.denominator)


def schedule_checks():
    t = F(1,3)
    lower = 2*sum((t**(2*n+1)/F(2*n+1) for n in range(96)),F(0))
    upper = lower+2*t**193/(193*(1-t*t))
    records = []
    for j in range(1,9):
        k = 2**j
        a = F(1,k**4)
        llow,lhigh = (2+2*j*lower)/a,(2+2*j*upper)/a
        assert ceil_rational(llow)==ceil_rational(lhigh)
        slow,shigh = 2*j*lower,2*j*upper
        assert ceil_rational(slow)==ceil_rational(shigh)
        records.append({'k':k,'a':str(a),'tau':str(a),'speed':'1',
                        'time_steps':k**4,'L':ceil_rational(llow),
                        'J':k**2,'K':k**2,'L_s':ceil_rational(slow),
                        'tail_budget':str(F(1,k*k)),
                        'step_and_generator_budget':str(a)})
    return {'classification':'CONDITIONAL_MATCHED_SPEED_SCHEDULE',
            'u':4,'v':4,'alpha':2,'beta':2,'alpha_K':1,'beta_K':1,
            's':1,'p':2,'mu_s':1,'c_L':2,'c_J':2,'c_K':2,'c_s':2,
            'reference_speed':'1','T':'1','log2_rational_terms':96,
            'records':records}


def other_checks():
    cone = [{'reference_speed':k,'other_speed':k,'speed_ratio':'1',
             'relative_cone_metric':'0'} for k in [1,2,4,8,16]]
    n = [F(1,3),F(2,3),F(2,3)]
    assert sum(x*x for x in n)==1
    quartic = sum(x**4 for x in n)
    assert quartic==F(11,27)
    # A scalar perturbation passes the necessary norm condition without
    # producing a large condition number.
    delta,target = F(1,4),F(3,2)
    assert delta>=(target-1)/(target+1)
    effective = identity(3)+delta*identity(3)
    singular_values = [abs(effective[i,i]) for i in range(3)]
    condition_number = max(singular_values)/min(singular_values)
    assert condition_number==1 and condition_number<target
    assert (1+delta)/(1-delta)>=target
    return {'classification':'FINITE_ALGEBRAIC_COMPARISONS',
            'equal_speed_sequences':cone,
            'quartic_axis_value':'1','quartic_other_direction_value':str(quartic),
            'persistent_difference_for_coefficient_a_to_minus2':str(1-quartic),
            'Schur_scalar_witness':{'M':'I3','Sigma':'I3/4','rho':'0',
                 'delta':str(delta),'target_condition_number':str(target),
                 'actual_condition_number':'1','necessary_threshold':str((target-1)/(target+1))}}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    moments = moment_tail_checks()
    schedule = schedule_checks()
    cert = {'classification':'FINITE_COMPARISONS_AND_CONDITIONAL_CONTINUUM_CONTROLS',
            'arithmetic':'RATIONAL_AND_BOUNDED_INTEGER',
            'program_sha256':sha(Path(__file__).resolve()),
            'environment':{'Python':platform.python_version(),'NumPy':np.__version__},
            'state_source_continuity':state_source_checks(),
            'CAR_and_Gram_support':car_checks(),'boundary_phase_selection':boundary_checks(),
            'finite_moment_tail':moments,'square_root_state_dependence':holder_check(),
            'matched_speed_schedule':schedule,'cone_and_Schur_controls':other_checks()}
    names=['sections/main/m120.tex','sections/main/m121.tex','sections/supplement/s73.tex',
           'sections/supplement/s75.tex','sections/supplement/s76.tex',
           'sections/supplement/s78.tex','sections/supplement/s144.tex']
    cert['proof_sources_sha256']={'paper/'+name:sha(ROOT/'paper'/name) for name in names}
    with (OUT/'MOMENT_TAIL_BOUNDS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(moments[0]));writer.writeheader();writer.writerows(moments)
    with (OUT/'MATCHED_SPEED_SCHEDULE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(schedule['records'][0]));writer.writeheader();writer.writerows(schedule['records'])
    (OUT/'CONTINUUM_AND_SOURCE_CERTIFICATE.json').write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps({'verification':'PASSED','rational_moment_checks':130,
                      'CAR_checks':72,'spin_configurations':512,
                      'classification':cert['classification']},indent=2))


if __name__=='__main__':
    main()
