#!/usr/bin/env python3
"""Wilson CAR modules and all-occupation static one-plaquette contractions.

The local coefficients, charge assignments, source duals and primitive seed
come from the existing QICT records. This is a gauge-only one-plaquette
calculation on the product cover; it does not instantiate the moving C10.
"""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import hashlib, json, math, platform
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reproducibility/results/static_wilson_car'
INPUT = ROOT/'reproducibility/results/nonabelian_common_car'
PHI = math.pi/63
DEPTH = 12
TIMES = 7
CUTOFFS = (24, 36, 48, 60)
CYCLIC_LENGTH = 504
WEAK_SEED_RADIUS = 18


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def creators(modes):
    result = []
    for a in range(modes):
        columns = [n for n in range(1 << modes) if not n & (1 << a)]
        values = [(-1)**((n & ((1 << a)-1)).bit_count()) for n in columns]
        result.append(sparse.csr_matrix((values, ([n | (1 << a) for n in columns], columns)),
                                       shape=(1 << modes, 1 << modes)))
    return result


def quaternion(q):
    a,b,c,d = map(float, q)
    return np.array([[a+1j*b, c+1j*d], [-c+1j*d, a-1j*b]])


def fock_wilson(u, theta):
    """Second quantization in the stored nine-component occupation basis."""
    local = np.zeros((4,4), complex)
    local[0,0] = 1
    local[1:3,1:3] = u
    local[3,3] = np.linalg.det(u)
    weak = sparse.csr_matrix([[1.]])
    for _ in range(3): weak = sparse.kron(local, weak, format='csr')
    weak = sparse.kron(sparse.eye(8), weak, format='csr')
    charge = np.array([-3*(n & 63).bit_count()-6*(n >> 6).bit_count() for n in range(512)])
    return sparse.diags(np.exp(1j*theta*charge), format='csr') @ weak


def undressed_receiver_module():
    """Each left occupation retains its two matter and dual indices."""
    rows,cols,values = [],[],[]
    for mask in range(64):
        occupied = [a for a in range(3) if mask & (1 << a)]
        for alpha in product((0,1), repeat=len(occupied)):
            fermion = sum(1 << (6+a) for a in range(3) if mask & (1 << (3+a)))
            dual = 0
            for a,b in zip(occupied,alpha):
                fermion |= 1 << (2*a+b)
                dual += (b+1)*3**a
            rows.append(512*dual+fermion);cols.append(mask)
            values.append(2**(-len(occupied)/2))
    return sparse.csr_matrix((values,(rows,cols)),shape=(512*27,64))


def dual_gauge(u, angle):
    local = np.zeros((3,3),complex);local[0,0]=1
    local[1:,1:] = np.exp(3j*angle)*u.conj()
    result=sparse.csr_matrix([[1.]])
    for _ in range(3):result=sparse.kron(local,result,format='csr')
    return result


def norm(a): return float(sparse.linalg.norm(a))


def car_modules(arrays):
    bare9,bare6 = creators(9),creators(6)
    identity9=sparse.eye(512,format='csr')
    module0=undressed_receiver_module()
    samples=[(F(1),F(0),F(0),F(0)),(F(1,2),)*4,
             (F(3,5),F(0),F(4,5),F(0)),(F(0),F(1),F(0),F(0))]
    records=[]
    for index,q in enumerate(samples):
        theta=(index-1)*PHI;u=quaternion(q)
        D=fock_wilson(u,theta)
        T=sparse.kron(sparse.eye(27),D,format='csr')@module0
        P=T@T.conj().T
        dressed=[D@c@D.conj().T for c in bare9]
        b=[T@c@T.conj().T for c in bare6]
        nine_error=six_error=parity_error=0.
        for i in range(9):
            for j in range(9):
                nine_error=max(nine_error,norm(dressed[i].conj().T@dressed[j]+
                    dressed[j]@dressed[i].conj().T-(identity9 if i==j else 0)))
                nine_error=max(nine_error,norm(dressed[i]@dressed[j]+dressed[j]@dressed[i]))
        for i in range(6):
            for j in range(6):
                six_error=max(six_error,norm(b[i].conj().T@b[j]+b[j]@b[i].conj().T-(P if i==j else 0)))
                six_error=max(six_error,norm(b[i]@b[j]+b[j]@b[i]))
        parity=sparse.diags([(-1)**n.bit_count() for n in range(512)],format='csr',dtype=np.int64)
        parity=sparse.kron(sparse.eye(27),parity,format='csr')
        for a in b:parity_error=max(parity_error,norm(parity@a+a@parity))
        hs=quaternion(samples[(index+1)%4]);hc=quaternion(samples[(index+2)%4])
        asource,acomp=3*PHI,-2*PHI
        pulled=sparse.kron(sparse.eye(27),fock_wilson(hs.conj().T@u@hc,
                                      theta-asource+acomp),format='csr')@module0
        matter=fock_wilson(hs,asource)
        right_counter=sparse.diags([np.exp(6j*acomp*(n>>6).bit_count()) for n in range(512)])
        transform=sparse.kron(dual_gauge(hc,acomp),right_counter@matter,format='csr')
        covariance=norm(transform@pulled-T)
        row={'quaternion_rational':[str(x) for x in q],'angle_multiplier_pi_over_63':index-1,
             'nine_component_CAR_Frobenius_diagnostic':nine_error,
             'six_receiver_module_CAR_Frobenius_diagnostic':six_error,
             'module_Gram_Frobenius_diagnostic':norm(T.conj().T@T-sparse.eye(64)),
             'endpoint_Gauss_covariance_diagnostic':covariance,'odd_parity_diagnostic':parity_error}
        assert max(nine_error,six_error,covariance,parity_error,row['module_Gram_Frobenius_diagnostic'])<5e-13
        for name,matrix in [('module',T),('fock_unitary',D)]:
            prefix=f'configuration{index}_{name}'
            arrays[prefix+'_data']=matrix.data;arrays[prefix+'_indices']=matrix.indices
            arrays[prefix+'_indptr']=matrix.indptr;arrays[prefix+'_shape']=np.array(matrix.shape)
        records.append(row)
    return records


@lru_cache(None)
def cg_square(k,h,s,m,alpha):
    """Factorial formula, all angular momenta doubled, evaluated as a fraction."""
    total_m=m+alpha
    if (abs(total_m)>s or abs(m)>k or abs(alpha)>h or s<abs(k-h) or s>k+h
        or (s+k+h)%2):return F(0)
    f=math.factorial
    a,b,c=(s+k-h)//2,(s-k+h)//2,(k+h-s)//2
    pref=F((s+1)*f(a)*f(b)*f(c),f((s+k+h)//2+1))
    for x in (s+total_m,s-total_m,k-m,k+m,h-alpha,h+alpha):pref*=f(x//2)
    value=F(0)
    for ell in range(min(c,(k-m)//2,(h+alpha)//2)+1):
        arguments=(ell,c-ell,(k-m)//2-ell,(h+alpha)//2-ell,
                   (s-h+m)//2+ell,(s-k-alpha)//2+ell)
        if min(arguments)>=0:value+=F((-1)**ell,math.prod(f(x) for x in arguments))
    return pref*value**2


def weak_labels(h,cutoff):
    return [(s,t) for s in range(cutoff+1) for t in range(abs(s-h),s+h+1,2)]


@lru_cache(None)
def polynomials(h,s,t):
    return tuple({m:cg_square(t,h,s,m,alpha) for m in range(-t,t+1,2)
                  if cg_square(t,h,s,m,alpha)} for alpha in range(-h,h+1,2))


def laurent_overlap(first,second,shifts):
    result=F(0)
    for p,q in zip(first,second):
        for m,value in q.items():
            for shift in shifts:
                result+=value*(p.get(m+shift,F(0))-
                    (p.get(m+shift+2,F(0))+p.get(m+shift-2,F(0)))/2)
    return result


def weak_matrices(h,cutoff):
    labels=weak_labels(h,cutoff);index={x:i for i,x in enumerate(labels)}
    rows,cols,values,symbolic=[],[],[],[]
    for j,(s,t) in enumerate(labels):
        diagonal=laurent_overlap(polynomials(h,s,t),polynomials(h,s,t),(0,))
        assert diagonal*F(t+1,s+1)==1
        for ss in (s-1,s+1):
            for tt in (t-1,t+1):
                if (ss,tt) not in index:continue
                i=index[(ss,tt)]
                coefficient=laurent_overlap(polynomials(h,ss,tt),polynomials(h,s,t),(-1,1))
                radicand=F((t+1)*(tt+1),(s+1)*(ss+1))
                if coefficient:
                    rows.append(i);cols.append(j);values.append(float(coefficient)*math.sqrt(float(radicand)))
                    symbolic.append([i,j,str(coefficient),str(radicand)])
    M=sparse.csr_matrix((values,(rows,cols)),shape=(len(labels),len(labels)))
    assert norm(M-M.T)<5e-14
    # All Gram entries in a low subspace are also checked with rational arithmetic.
    low=weak_labels(h,min(8,cutoff))
    for a in low:
        for b in low:
            value=laurent_overlap(polynomials(h,*a),polynomials(h,*b),(0,))
            assert value==(F(a[0]+1,a[1]+1) if a==b else 0)
    source=sparse.csr_matrix(([math.sqrt(F(t+1,(h+1)*(s+1))) for s,t in labels],
             (range(len(labels)),[s for s,t in labels])),shape=(len(labels),cutoff+1))
    assert norm(source.T@source-sparse.eye(cutoff+1))<5e-14
    E=np.array([3*s*(s+2)+t*(t+2) for s,t in labels],np.int64)
    return labels,M,E,source,symbolic


def weak_apply(M,E,vector,inverse=False):
    diagonal=np.exp(-1j*PHI*(E%126))
    if inverse:return diagonal.conj()*np.exp(32j*PHI)*expm_multiply(-16j*PHI*M,vector,traceA=0)
    return np.exp(-32j*PHI)*expm_multiply(16j*PHI*M,diagonal*vector,traceA=0)


def y_apply(Q,vector,inverse=False):
    length=len(vector);r=np.arange(length)-length//2
    E=(8*r*r+4*Q*r+2*Q*Q+Q)%126
    diagonal=np.exp(-1j*PHI*E)
    magnetic=np.exp(-1j*PHI*(4-4*np.cos(2*np.pi*np.arange(length)/length)))
    if inverse:return diagonal.conj()*np.fft.ifft(magnetic.conj()*np.fft.fft(vector))
    return np.fft.ifft(magnetic*np.fft.fft(diagonal*vector))


def orbit(seed,apply,depth):
    states=[seed.astype(complex)]
    for _ in range(depth):states.append(apply(states[-1]))
    return np.asarray(states)


def returns(seed,apply):
    positive=orbit(seed,apply,DEPTH)
    return np.einsum('j,nj->n',seed.conj(),positive),positive


def inverse_fields(neutral_states,source,apply_inverse):
    fields=[]
    for s,vector in enumerate(neutral_states[:TIMES]):
        vector=source@vector
        for _ in range(s):vector=apply_inverse(vector)
        fields.append(vector)
    fields=np.asarray(fields)
    return fields,fields.conj()@fields.T


def arctan_interval(x,terms):
    partial=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(terms)),F(0))
    next_term=(-1)**terms*x**(2*terms+1)/F(2*terms+1)
    return min(partial,partial+next_term),max(partial,partial+next_term)


def pi_interval(terms=14):
    a,b=arctan_interval(F(1,5),terms);c,d=arctan_interval(F(1,239),terms)
    return 16*a-4*d,16*b-4*c


def sine_square_interval(multiplier,pi_bounds):
    lo,hi=(multiplier*x for x in pi_bounds)
    degree=51
    value=sum(((-1)**k*lo**(2*k+1)/math.factorial(2*k+1) for k in range((degree+1)//2)),F(0))
    error=abs(lo)**(degree+1)/math.factorial(degree+1)+hi-lo
    a,b=value-error,value+error
    return (F(0) if a<=0<=b else min(a*a,b*b)),max(a*a,b*b)


def outward(value,up=False,denominator=10**16):
    scaled=value*denominator
    integer=scaled.numerator//scaled.denominator
    if up and scaled.denominator!=1:integer+=1
    return F(integer,denominator)


def sqrt_upper(value,digits=50):
    denominator=10**digits
    numerator=math.isqrt(value.numerator*denominator**2//value.denominator)
    if F(numerator,denominator)**2<value:numerator+=1
    return F(numerator,denominator)


def leakage_certificate(ground):
    witness=[F(x) for x in ground['rational_witness_components']]
    normalization=sum(x*x for x in witness)
    distance=F(ground['state_trace_distance_upper_rational'])
    bounds=pi_interval();lower=upper=F(0)
    for s,amplitude in enumerate(witness):
        a,b=sine_square_interval(F(2*(s+1),63),bounds)
        weight=amplitude**2/normalization*F(s*(s+2),(s+1)**2)
        lower+=weight*a;upper+=weight*b
    lower,upper=outward(lower-distance),outward(upper+distance,True)
    pure_a,pure_b=sine_square_interval(F(4,63),bounds)
    return {'pi_enclosure_rational':[str(x) for x in bounds],
        'primitive_ground_squared_leakage_interval_rational':[str(lower),str(upper)],
        'primitive_ground_squared_leakage_interval':[float(lower),float(upper)],
        'j_half_squared_leakage_interval_rational':[str(outward(F(3,4)*pure_a)),str(outward(F(3,4)*pure_b,True))],
        'j_half_exact_expression':'(3/4)*sin(4*pi/63)^2',
        'primitive_trace_distance_used_rational':str(distance),
        'scope':'Leakage from the Wilson image under the actual isolated weak electric gate; magnetic multiplication preserves that image.'}


def tensor_weights(nleft):
    return {0:{0:F(1)},1:{1:F(1)},2:{0:F(1,4),2:F(3,4)},
            3:{1:F(1,2),3:F(1,2)}}[nleft]


def return_tail(cutoff,seed_radius,steps,x):
    degree=2*(cutoff+1-seed_radius)
    radius=F(degree,steps)/x
    if radius<=1:return F(2)
    return min(F(2),2*(F(11,4)*steps*x/degree)**degree)


def contractions(primitive,arrays):
    grounds=primitive['cutoff_results'][-1]['ground_certificates']
    weak_full=[F(x) for x in grounds['SU2']['rational_witness_components']]
    norm_full=sum(x*x for x in weak_full)
    weak_core=weak_full[:WEAK_SEED_RADIUS+1]
    discarded=sum(x*x for x in weak_full[WEAK_SEED_RADIUS+1:])/norm_full
    seed_distance=sqrt_upper(discarded)
    weak=np.array([float(x) for x in weak_core])
    weak/=np.linalg.norm(weak)
    ysmall=np.array([float(F(x)) for x in grounds['U1']['rational_witness_components']])
    ysmall/=np.linalg.norm(ysmall)
    seedY=np.zeros(CYCLIC_LENGTH,complex);rY=(len(ysmall)-1)//2
    seedY[CYCLIC_LENGTH//2-rY:CYCLIC_LENGTH//2+rY+1]=ysmall
    y0,y0states=returns(seedY,lambda v:y_apply(0,v))
    ycharged={};yfields={};ykernels={}
    for q in sorted({-3*nl-6*nr for nl in range(4) for nr in range(4)}):
        ret,_=returns(seedY,lambda v,q=q:y_apply(q,v));ycharged[q]=ret
        ff,kk=inverse_fields(y0states,sparse.eye(CYCLIC_LENGTH),lambda v,q=q:y_apply(q,v,True))
        yfields[q]=ff;ykernels[q]=kk
    with np.load(INPUT/'NONABELIAN_COMMON_CAR_KATO_MOMENTS.npz',allow_pickle=False) as prior:
        colour=prior['L3_neutral_group_returns'][2]
        six_before=prior['L3_gauge_Cn6']
    records=[];previous=None
    for cutoff in CUTOFFS:
        seedW=np.zeros(cutoff+1,complex);seedW[:len(weak)]=weak
        blocks={};weak_returns={};weak_kernels={}
        for h in range(4):
            labels,M,E,T,symbolic=weak_matrices(h,cutoff)
            blocks[h]=(M,E,T)
            if h==0:
                w0,w0states=returns(seedW,lambda v:weak_apply(M,E,v))
            ret,_=returns(T@seedW,lambda v,M=M,E=E:weak_apply(M,E,v))
            weak_returns[h]=ret
            ff,kk=inverse_fields(w0states,T,lambda v,M=M,E=E:weak_apply(M,E,v,True))
            weak_kernels[h]=kk
            if cutoff==CUTOFFS[-1]:
                prefix=f'weak_J{h}'
                arrays[prefix+'_labels']=np.array(labels,np.int64)
                arrays[prefix+'_magnetic_data']=M.data;arrays[prefix+'_magnetic_indices']=M.indices
                arrays[prefix+'_magnetic_indptr']=M.indptr
                arrays[prefix+'_source_data']=T.data;arrays[prefix+'_source_indices']=T.indices
                arrays[prefix+'_source_indptr']=T.indptr;arrays[prefix+'_electric']=E
                arrays[prefix+'_symbolic_edges']=np.array(symbolic,dtype=str)
        diagonal=np.zeros((DEPTH+1,64),complex);kernels=np.zeros((64,TIMES,TIMES),complex)
        negative=np.zeros_like(diagonal)
        for mask in range(64):
            nl=(mask&7).bit_count();nr=(mask>>3).bit_count();q=-3*nl-6*nr
            wr=sum(float(weight)*weak_returns[h] for h,weight in tensor_weights(nl).items())
            wk=sum(float(weight)*weak_kernels[h] for h,weight in tensor_weights(nl).items())
            diagonal[:,mask]=y0.conj()*ycharged[q]*w0.conj()*wr*abs(colour)**2
            kernels[mask]=ykernels[q]*wk
        # Inverse propagation is run separately at the final box.
        if cutoff==CUTOFFS[-1]:
            yn={q:returns(seedY,lambda v,q=q:y_apply(q,v,True))[0] for q in ycharged}
            wn={h:returns(T@seedW,lambda v,M=M,E=E:weak_apply(M,E,v,True))[0]
                for h,(M,E,T) in blocks.items()}
            for mask in range(64):
                nl=(mask&7).bit_count();nr=(mask>>3).bit_count();q=-3*nl-6*nr
                wr=sum(float(weight)*wn[h] for h,weight in tensor_weights(nl).items())
                negative[:,mask]=yn[0].conj()*yn[q]*wn[0].conj()*wr*abs(colour)**2
            arrays['negative_comparison_moment_diagonal']=negative
            arrays['neutral_U1_states']=y0states;arrays['neutral_weak_states']=w0states
            arrays['neutral_colour_returns']=colour
        toeplitz=[];gram=[]
        for mask in range(64):
            c=diagonal[:,mask]
            t=np.array([[c[j-i] if j>=i else c[i-j].conj() for j in range(13)] for i in range(13)])
            toeplitz.append(float(np.linalg.eigvalsh(t).min()))
            gram.append(float(np.linalg.eigvalsh(kernels[mask]).min()))
        radius=len(weak)-1
        tail=return_tail(cutoff,radius,12,F(8,5))
        Dy,Dw,Dc=(F(grounds[g]['state_trace_distance_upper_rational']) for g in ('U1','SU2','SU3'))
        Dw+=seed_distance
        alias_degree=CYCLIC_LENGTH-2*rY
        ytail=min(F(2),2*(F(11,4)*12*F(1,5)/alias_degree)**alias_degree)
        colour_tail=F(primitive['analytic_polynomial_tail_bounds_rational']['SU3'])
        colour_return=2*12*colour_tail*(1+colour_tail)**11
        comparison_bound=min(F(2),4*(Dy+Dw+Dc)+4*(tail+ytail+colour_return))
        physical_bound=min(F(2),2*(Dy+Dw)+2*(tail+ytail))
        row={'twice_spin_box':cutoff,'weak_dimensions':{str(h):len(weak_labels(h,cutoff)) for h in range(4)},
             'C0_maximum_diagnostic':float(np.max(abs(diagonal[0]-1))),
             'comparison_Toeplitz_minimum_diagnostic':min(toeplitz),
             'physical_Gram_minimum_diagnostic':min(gram),
             'physical_equal_time_maximum_diagnostic':float(np.max(abs(np.diagonal(kernels,axis1=1,axis2=2)-1))),
             'successive_comparison_moment_change_diagnostic':None if previous is None else float(np.max(abs(diagonal-previous))),
             'closed_return_error_bound_rational':str(outward(tail,True)),
             'comparison_moment_error_bound_rational':str(outward(comparison_bound,True)),
             'physical_kernel_error_bound_rational':str(outward(physical_bound,True))}
        assert min(toeplitz)>-2e-12 and min(gram)>-2e-12
        assert row['C0_maximum_diagnostic']<2e-12 and row['physical_equal_time_maximum_diagnostic']<2e-12
        arrays[f'R{cutoff}_comparison_moment_diagonal']=diagonal
        arrays[f'R{cutoff}_physical_kernels']=kernels
        records.append(row);previous=diagonal
        print('completed static occupation box',cutoff,flush=True)
    sample=diagonal[:,[1,2,4,8,16,32]]
    before=np.diagonal(six_before,axis1=1,axis2=2)
    existing_error=float(np.max(abs(sample-before)))
    assert existing_error<2e-11
    assert np.max(abs(kernels[0]-1))<2e-12
    inverse_error=float(np.max(abs(negative-diagonal.conj())))
    assert inverse_error<2e-12
    return records,{'six_existing_sources_comparison_diagnostic':existing_error,
                    'inverse_moment_diagnostic':inverse_error,
                    'empty_source_physical_kernel_identity_diagnostic':float(np.max(abs(kernels[0]-1))),
                    'weak_seed_core_radius':WEAK_SEED_RADIUS,
                    'weak_seed_discarded_norm_squared_rational':str(discarded),
                    'weak_seed_trace_distance_upper_rational':str(seed_distance)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    primitive_path=INPUT/'NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json'
    primitive=json.loads(primitive_path.read_text())
    arrays={};module_records=car_modules(arrays)
    cutoff_records,diagnostics=contractions(primitive,arrays)
    leakage=leakage_certificate(primitive['cutoff_results'][-1]['ground_certificates']['SU2'])
    data=OUT/'STATIC_WILSON_CAR_MODULES_AND_KERNELS.npz'
    np.savez_compressed(data,**arrays)
    certificate={'status':'PASS_STATIC_WILSON_CAR_AND_OCCUPATION_KERNELS',
       'source_program_sha256':sha(Path(__file__)),
       'input_certificate_sha256':sha(primitive_path),
       'input_moments_sha256':sha(INPUT/'NONABELIAN_COMMON_CAR_KATO_MOMENTS.npz'),
       'data_sha256':sha(data),'actual_plaquette_edges':primitive['actual_plaquette_edges'],
       'source_order':['L1','L2','L3','R1','R2','R3'],
       'canonical_component_modes':9,'component_Fock_dimension':512,
       'receiver_occupation_module_dimension':64,'dual_dimension':27,
       'endpoint_singlet_maps':['vacuum plus dual doublet per left family','tied singlet countercharge per right family'],
       'source_module_CAR_identity':'P_module; I only on the module, not the ambient field Hilbert space',
       'source_module_dynamically_invariant':False,
       'module_checks':module_records,'four_representation_boxes':cutoff_records,
       'irreducible_twice_spins':[0,1,2,3],
       'physical_time_range':[0,6],'comparison_moment_depth':12,
       'occupation_charge_range':[-36,0],
       'spin_decomposition_scope':'Pure tensor-product singlets with multiplicity spaces retained; return sums follow block orthogonality, not dephasing.',
       'leakage':leakage,'diagnostics':diagnostics,
       'common_primitive_state':'Existing three-group one-plaquette A7 neutral state; empty source reference.',
       'spatial_convergence_proved':False,'full_C10_contract_completed':False,
       'new_physical_parameters':0,'floating_point_interval_certified':False,
       'arithmetic':'Fraction CG squares, rational state and trigonometric bounds; sparse and FFT evaluations are floating-point diagnostics.',
       'environment':{'Python':platform.python_version(),'NumPy':np.__version__,'SciPy':scipy.__version__}}
    (OUT/'STATIC_WILSON_CAR_CERTIFICATE.json').write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps({'status':certificate['status'],'occupation_sectors':64,
                     'leakage_interval':leakage['primitive_ground_squared_leakage_interval'],
                     'diagnostics':diagnostics},indent=2))


if __name__=='__main__':main()
