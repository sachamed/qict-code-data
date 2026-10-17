#!/usr/bin/env python3
"""Independent propagation and exact-data checks for source memory.

Magnetic exponentials use Hermitian diagonalization. Abelian propagation
uses a Bessel shift kernel. Complement vectors are propagated directly in
the original charged basis, without the producer's complement frame.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib, io, json, math
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.special import jv
from prepare_inputs import read_reference

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/coherent_source_memory'
SOURCE=ROOT/'reproducibility/results/static_wilson_car'
PHI=math.pi/63


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def y_step(vector,Q,inverse=False):
    r=np.arange(504)-252;cost=(8*r*r+4*Q*r+2*Q*Q+Q)%126
    angle=4*PHI
    if not inverse:vector=np.exp(-1j*PHI*cost)*vector
    direction=-1 if inverse else 1
    magnetic=sum((direction*1j)**d*jv(d,angle)*np.roll(vector,d) for d in range(-18,19))
    magnetic*=np.exp(-direction*1j*angle)
    return np.exp(1j*PHI*cost)*magnetic if inverse else magnetic


def neutral_orbit(seed,apply):
    result=[seed.copy()]
    for _ in range(6):result.append(apply(result[-1]))
    return result


def elementary_cos(cost):
    """Independent rational Taylor enclosure, including a Machin interval."""
    def atan(q):
        x=F(1,q);a=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(36)),F(0))
        return a,a+x**73/F(73)
    a,b=atan(5);c,d=atan(239);lo=16*a-4*d;hi=16*b-4*c
    residue=(int(cost)+63)%126-63
    x=F(residue,126)*(lo+hi)
    value=sum(((-1)**k*x**(2*k)/math.factorial(2*k) for k in range(40)),F(0))
    error=F(abs(residue),126)*(hi-lo)+F(22,7)**80/math.factorial(80)
    return value-error,value+error


def exact_faces():
    content=json.loads((ROOT/'reproducibility/verification/SCIENTIFIC_CONTENT.json').read_text())
    name='reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    with np.load(io.BytesIO(read_reference(content['logical_scientific_files'][name])),allow_pickle=False) as z:
        links=z['links'].astype(int);start=links[:,1:];end=start.copy()
        end[np.arange(len(links)),links[:,0]]+=1
        vertices,indices=np.unique(np.concatenate([start,end]),axis=0,return_inverse=True)
        parent=list(range(len(vertices)))
        def find(a):
            while parent[a]!=a:
                parent[a]=parent[parent[a]];a=parent[a]
            return a
        for a,b in zip(indices[:len(links)],indices[len(links):]):
            parent[find(int(a))]=find(int(b))
        assert len({find(i) for i in range(len(vertices))})==1
        for edges,signs in zip(z['plaquette_edge_indices'],z['plaquette_edge_sign']):
            forward=[tuple(start[e] if k>0 else end[e]) for e,k in zip(edges,signs)]
            backward=[tuple(end[e] if k>0 else start[e]) for e,k in zip(edges,signs)]
            assert len(set(forward))==len(set(backward))==4
            assert sorted(forward)==sorted(backward)
            assert np.array_equal(sum(k*(end[e]-start[e]) for e,k in zip(edges,signs)),np.zeros(3,dtype=int))
    return 18252


def rational_stationarity():
    """Separate Fraction polynomial of the neutral character Jacobi update."""
    def rounded(x,up,base=10**50):
        q=x*base;n=q.numerator//q.denominator
        if up and q.denominator!=1:n+=1
        return F(n,base)
    def atan(q):
        x=F(1,q);a=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(36)),F(0))
        return a,a+x**73/F(73)
    a,b=atan(5);cc,dd=atan(239)
    lo,hi=rounded(16*a-4*dd,False),rounded(16*b-4*cc,True)
    pi=(lo+hi)/2;pi_error=(hi-lo)/2
    ground=json.loads((ROOT/'reproducibility/results/nonabelian_common_car/NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json').read_text())['cutoff_results'][-1]['ground_certificates']['SU2']
    source=json.loads((SOURCE/'STATIC_WILSON_CAR_CERTIFICATE.json').read_text())
    radius=source['diagnostics']['weak_seed_core_radius']
    w=[F(x) for x in ground['rational_witness_components'][:radius+1]]
    w+= [F(0)]*(61-len(w));normalization=sum(x*x for x in w)
    def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
    state=[]
    for s,v in enumerate(w):
        cost=(4*s*(s+2)+32+63)%126-63
        term=(F(1),F(0));total=term
        for k in range(1,41):
            x,y=mul(term,(F(0),-cost*pi/63));term=(x/k,y/k)
            total=(total[0]+term[0],total[1]+term[1])
        state.append((v*total[0],v*total[1]))
    term=state;result=list(state);lambda_m=16*pi/63
    for k in range(1,41):
        new=[]
        for s in range(61):
            neighbours=([term[s-1]] if s else [])+([term[s+1]] if s<60 else [])
            real=sum(t[0] for t in neighbours);imag=sum(t[1] for t in neighbours)
            new.append((-lambda_m*imag/k,lambda_m*real/k))
        term=new;result=[(a+c,b+d) for (a,b),(c,d) in zip(result,term)]
    real=sum(v*t[0] for v,t in zip(w,result))/normalization
    imag=sum(v*t[1] for v,t in zip(w,result))/normalization
    def tail(x):return x**41/math.factorial(41)/(1-x/42)
    electric_tail=tail(F(22,7));magnetic_tail=tail(F(8,5))
    error=electric_tail+magnetic_tail*(1+electric_tail)+pi_error*(2+electric_tail+magnetic_tail)
    error+=2*(F(ground['state_trace_distance_upper_rational'])+F(source['diagnostics']['weak_seed_trace_distance_upper_rational']))
    centre=1-real*real-imag*imag;deviation=2*error+3*error**2
    interval=[rounded(centre-deviation,False,10**30),rounded(centre+deviation,True,10**30)]
    return interval


def main():
    cp=OUT/'COHERENT_SOURCE_MEMORY_CERTIFICATE.json';c=json.loads(cp.read_text())
    assert c['status']=='PASS_COHERENT_SOURCE_MEMORY_AND_CENTRAL_LIFT'
    data=OUT/'COHERENT_SOURCE_MEMORY_DATA.npz';assert sha(data)==c['data_sha256']
    assert sha(ROOT/'reproducibility/code/compute_coherent_source_memory.py')==c['source_program_sha256']
    predecessor=SOURCE/'STATIC_PHYSICAL_KERNEL_ARITHMETIC_CERTIFICATE.json'
    p=json.loads(predecessor.read_text());assert sha(predecessor)==c['preceding_kernel_arithmetic_certificate_sha256']
    records=[];weak={};maximum=0.;unitaries={};selectors=[];phase_aliases=[]
    with np.load(SOURCE/'STATIC_WILSON_CAR_MODULES_AND_KERNELS.npz',allow_pickle=False) as z,np.load(data,allow_pickle=False) as d:
        for h in range(4):
            labels=z[f'weak_J{h}_labels'];dim=len(labels)
            M=sparse.csr_matrix((z[f'weak_J{h}_magnetic_data'],z[f'weak_J{h}_magnetic_indices'],z[f'weak_J{h}_magnetic_indptr']),shape=(dim,dim)).toarray()
            T=sparse.csr_matrix((z[f'weak_J{h}_source_data'],z[f'weak_J{h}_source_indices'],z[f'weak_J{h}_source_indptr']),shape=(dim,61)).toarray()
            for s in range(61):
                rows=np.flatnonzero(labels[:,0]==s)
                costs=[int(z[f'weak_J{h}_electric'][i]) for i in rows]
                assert len(set(costs))==len(costs)
                for a,e in enumerate(costs):
                    for b,argument in enumerate(costs):
                        value=F(1)
                        for k,other in enumerate(costs):
                            if k!=a:value*=F(argument-other,e-other)
                        assert value==int(a==b)
                        if b>a and (e-argument)%126==0:
                            phase_aliases.append({'twice_spin':h,'outer_twice_spin':s,
                                'shifted_twice_spins':[int(labels[rows[a],1]),int(labels[rows[b],1])],
                                'electric_cost_difference':abs(e-argument)})
                selectors.append(len(costs))
            energy,rotation=eigh(M)
            U=(rotation*np.exp(16j*PHI*energy))@rotation.T
            U*=np.exp(-32j*PHI)
            U*=np.exp(-1j*PHI*(z[f'weak_J{h}_electric']%126))[None,:]
            unitaries[h]=U
            A=T.T@U@T
            x=np.eye(61,dtype=complex);q=np.zeros((dim,61),complex);result=[]
            for n in range(13):
                result.append(x.copy())
                following=U@(T@x+q)
                x=T.T@following;q=following-T@x
            result=np.asarray(result)
            error=float(np.max(abs(result-d[f'J{h}_source_returns'])))
            assert error<2e-13
            maximum=max(maximum,error)
            neutral=neutral_orbit(z['neutral_weak_states'][0],lambda v:unitaries[0]@v)
            kernel=np.zeros((7,7),complex)
            for t in range(7):
                for s in range(7):
                    R=result[t-s] if t>=s else result[s-t].conj().T
                    kernel[t,s]=np.vdot(neutral[t],R@neutral[s])
            weak[h]=kernel
            eigenvalues=np.linalg.eigvalsh(np.eye(61)-A.conj().T@A)
            assert eigenvalues.min()>-2e-13
            records.append({'twice_spin':h,'projected_complement_recurrence_maximum_entry_diagnostic':error,
                'positive_defect_minimum_eigenvalue_diagnostic':float(eigenvalues.min()),
                'maximum_defect_eigenvalue_diagnostic':float(eigenvalues.max())})
        yneutral=neutral_orbit(z['neutral_U1_states'][0],lambda v:y_step(v,0))
        Y={}
        for charge in range(-36,1,3):
            fields=[]
            for t,seed in enumerate(yneutral):
                v=seed.copy()
                for _ in range(t):v=y_step(v,charge,True)
                fields.append(v)
            fields=np.asarray(fields);Y[charge]=fields.conj()@fields.T
        weights={0:{0:1},1:{1:1},2:{0:.25,2:.75},3:{1:.5,3:.5}}
        all_kernels=[]
        for mask in range(64):
            nl=(mask&7).bit_count();nr=(mask>>3).bit_count()
            all_kernels.append(Y[-3*nl-6*nr]*sum(w*weak[h] for h,w in weights[nl].items()))
        kernels=np.asarray(all_kernels)
        kernel_difference=float(np.max(abs(kernels-d['physical_kernels_coherent_memory'])))
        assert kernel_difference<2e-13
        costs=c['central_lift']['central_costs']
        low=high=F(0)
        for a in costs:
            for b in costs:
                x,y=elementary_cos(a-b);low+=x/F(36);high+=y/F(36)
        quoted=[F(x) for x in c['central_lift']['trivial_sector_loss_squared_interval_rational']]
        assert quoted[0]<=1-high<=1-low<=quoted[1]
        centre_file=SOURCE/'STATIC_PHYSICAL_KERNEL_INTERVAL_CENTRES.npz'
        assert sha(centre_file)==c['interval_centre_data_sha256']
        maximum_enclosure=F(0)
        radius=F(p['maximum_finite_kernel_arithmetic_radius_rational'])+F(p['analytic_state_and_representation_bound_rational'])
        with np.load(centre_file,allow_pickle=False) as centres:
            denominator=int(str(centres['common_denominator']))
            for mask in range(64):
                nl=(mask&7).bit_count();nr=(mask>>3).bit_count()
                for t in range(7):
                    for s in range(7):
                        a,b=[F(int(v),denominator) for v in centres['gaussian_integer_centres'][4*nl+nr,t,s]]
                        value=d['physical_kernels_coherent_memory'][mask,t,s]
                        maximum_enclosure=max(maximum_enclosure,abs(F(float(value.real))-a)+abs(F(float(value.imag))-b)+2*radius)
        assert maximum_enclosure<=F(c['reported_memory_kernel_entry_error_upper_rational'])
        assert 7*maximum_enclosure<=F(c['reported_memory_kernel_seven_time_operator_error_upper_rational'])
        source=z['weak_J1_source_data']
        # Finite one-left complement singular values have the analytic two-branch spectrum.
        analytic=np.array([0]+[(1-1/(s+1)**2)*math.sin(2*(s+1)*PHI)**2 for s in range(1,61)])
        T=sparse.csr_matrix((source,z['weak_J1_source_indices'],z['weak_J1_source_indptr']),shape=(121,61)).toarray()
        A=T.T@unitaries[1]@T
        actual=np.linalg.eigvalsh(np.eye(61)-A.conj().T@A)
        defect_eigen_error=float(np.max(abs(np.sort(analytic)-actual)))
        assert defect_eigen_error<2e-13
    faces=exact_faces()
    central_neutral=0
    for mask in range(64):
        nl=(mask&7).bit_count();nr=(mask>>3).bit_count();charge=-3*nl-6*nr
        assert (charge+3*nl)%6==0
        central_neutral+=1
    multiplets=[(1,1,1),(1,0,4),(1,0,-2),(0,1,-3),(0,0,-6),(0,0,0)]
    central_kernel=[(a,b,n) for a in range(3) for b in range(2) for n in range(6)
        if all((2*a*t+3*b*w+n*q)%6==0 for t,w,q in multiplets)]
    assert sorted(central_kernel)==sorted((n%3,n%2,n) for n in range(6))
    state_interval=rational_stationarity()
    quoted=[F(v) for v in c['primitive_stationarity']['weak_primitive_trace_distance_squared_interval_rational']]
    assert quoted[0]<=state_interval[0]<=state_interval[1]<=quoted[1]
    certificate={'status':'PASS_INDEPENDENT_COHERENT_SOURCE_MEMORY',
        'source_program_sha256':sha(Path(__file__)),'producer_certificate_sha256':sha(cp),'producer_data_sha256':sha(data),
        'weak_projected_propagation_checks':records,'maximum_recurrence_entry_diagnostic':maximum,
        'all_64_kernel_maximum_difference_diagnostic':kernel_difference,
        'one_left_defect_spectrum_maximum_difference_diagnostic':defect_eigen_error,
        'integer_centre_enclosures_rechecked':True,'central_orbit_loss_rational_enclosure_rechecked':True,
        'primitive_stationarity_squared_independent_fraction_interval_rational':[str(v) for v in state_interval],
        'primitive_electric_selector_blocks_verified_exact':len(selectors),
        'primitive_electric_selectors_recover_charged_directions':sum(selectors),
        'finite_Floquet_branch_aliases':phase_aliases,
        'archived_face_oriented_cycles_checked':faces,
        'receiver_occupation_central_neutrality_checked_exact':central_neutral,
        'full_matter_central_kernel_checked_exact':[list(v) for v in central_kernel],
        'archived_gauge_graph_connected':True,
        'Bessel_propagation_bandwidth':18,'full_moving_C10_closure_inferred':False}
    (OUT/'INDEPENDENT_COHERENT_SOURCE_MEMORY_CERTIFICATE.json').write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps(certificate,indent=2))


if __name__=='__main__':main()
