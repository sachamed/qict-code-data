#!/usr/bin/env python3
"""Independent exterior-power, spin-addition and dense-kernel verification."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from functools import lru_cache
import hashlib, json, math
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,expm
from scipy.sparse.linalg import expm_multiply

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/static_wilson_car'
PHI=math.pi/63


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def csr(z,prefix):
    return sparse.csr_matrix((z[prefix+'_data'],z[prefix+'_indices'],z[prefix+'_indptr']),
                            shape=tuple(z[prefix+'_shape']))


def exterior_power(M):
    occupied=[[a for a in range(9) if n&(1<<a)] for n in range(512)]
    result=np.zeros((512,512),complex)
    for i,I in enumerate(occupied):
        for j,J in enumerate(occupied):
            if len(I)==len(J):result[i,j]=np.linalg.det(M[np.ix_(I,J)]) if I else 1
    return result


def direct_module(U,theta):
    rows,cols,values=[],[],[]
    for mask in range(64):
        active=[a for a in range(3) if mask&(1<<a)]
        phase=np.exp(-1j*theta*(3*len(active)+6*(mask>>3).bit_count()))
        for alpha in product((0,1),repeat=len(active)):
            for beta in product((0,1),repeat=len(active)):
                fermion=sum(1<<(6+a) for a in range(3) if mask&(1<<(3+a)))
                dual=0;value=phase/2**(len(active)/2)
                for a,x,y in zip(active,alpha,beta):
                    fermion|=1<<(2*a+x);dual+=(y+1)*3**a;value*=U[x,y]
                rows.append(512*dual+fermion);cols.append(mask);values.append(value)
    return sparse.csr_matrix((values,(rows,cols)),shape=(512*27,64))


@lru_cache(None)
def coupled_square(k,h,s,m,alpha):
    """Diagonalize J_total^2 in a fixed magnetic subspace of dimension <=4."""
    total=m+alpha
    states=[(M,total-M) for M in range(-k,k+1,2) if abs(total-M)<=h and (h-total+M)%2==0]
    if (m,alpha) not in states or abs(total)>s:return 0.
    index={x:i for i,x in enumerate(states)}
    K,H=k/2,h/2
    matrix=np.zeros((len(states),len(states)))
    for i,(M,A) in enumerate(states):
        mm,aa=M/2,A/2
        matrix[i,i]=K*(K+1)+H*(H+1)+2*mm*aa
        target=(M+2,A-2)
        if target in index:
            v=math.sqrt(max(0.,(K*(K+1)-mm*(mm+1))*(H*(H+1)-aa*(aa-1))))
            matrix[i,index[target]]=matrix[index[target],i]=v
    values,vectors=eigh(matrix)
    target=(s/2)*(s/2+1);where=int(np.argmin(abs(values-target)))
    assert abs(values[where]-target)<2e-10
    return float(vectors[index[(m,alpha)],where]**2)


def quadrature_block(h,cutoff=24):
    labels=[(s,t) for s in range(cutoff+1) for t in range(abs(s-h),s+h+1,2)]
    nq=160;theta=np.pi*np.arange(1,nq+1)/(nq+1)
    weights=2*np.sin(theta)**2/(nq+1)
    functions=np.zeros((len(labels),nq,h+1),complex)
    for i,(s,t) in enumerate(labels):
        for a,alpha in enumerate(range(-h,h+1,2)):
            for m in range(-t,t+1,2):
                functions[i,:,a]+=coupled_square(t,h,s,m,alpha)*np.exp(1j*m*theta)
        functions[i]*=math.sqrt((t+1)/(s+1))
    G=np.einsum('ita,jta,t->ij',functions.conj(),functions,weights,optimize=True)
    M=np.einsum('ita,jta,t->ij',functions.conj(),functions,weights*2*np.cos(theta),optimize=True)
    M=M.real
    gram_error=float(np.linalg.norm(G-np.eye(len(labels)),2))
    assert gram_error<4e-12
    T=np.zeros((len(labels),cutoff+1))
    source_error=0.
    for s in range(cutoff+1):
        character=np.sin((s+1)*theta)/np.sin(theta)
        source=character[:,None]*np.exp(-1j*theta[:,None]*np.arange(-h,h+1,2)[None,:])/math.sqrt(h+1)
        overlap=np.einsum('ita,ta,t->i',functions.conj(),source,weights)
        expected=np.array([math.sqrt((t+1)/((h+1)*(s+1))) if r==s else 0 for r,t in labels])
        source_error=max(source_error,float(np.linalg.norm(overlap-expected)))
        T[:,s]=overlap.real
    assert source_error<4e-12
    E=np.array([4*(3*(s/2)*(s/2+1)+(t/2)*(t/2+1)) for s,t in labels])
    U=np.exp(-32j*PHI)*expm(16j*PHI*M)@np.diag(np.exp(-1j*PHI*E))
    return labels,M,T,U,gram_error,source_error


def tensor_irreps(n):
    if not n:return {0:1.}
    X=np.array([[0,1],[1,0]],complex)/2
    Y=np.array([[0,-1j],[1j,0]],complex)/2
    Z=np.diag([1.,-1.])/2
    components=[]
    for sigma in (X,Y,Z):
        total=np.zeros((2**n,2**n),complex)
        for a in range(n):
            term=np.array([[1.]])
            for b in range(n):term=np.kron(term,sigma if a==b else np.eye(2))
            total+=term
        components.append(total)
    eigenvalues=np.linalg.eigvalsh(sum(A@A for A in components))
    result={}
    for h in range(n%2,n+1,2):
        J=h/2;count=int(np.sum(abs(eigenvalues-J*(J+1))<1e-12))
        if count:result[h]=count/2**n
    assert abs(sum(result.values())-1)<1e-15
    return result


def y_step(Q,v,inverse=False):
    length=len(v);r=np.arange(length)-length//2
    rows=np.arange(length);S=sparse.csr_matrix((np.ones(length),(rows,(rows-1)%length)),shape=(length,length))
    D=np.exp(-1j*PHI*(8*r*r+4*Q*r+2*Q*Q+Q))
    magnetic=2j*PHI*(S+S.T)
    if inverse:return D.conj()*np.exp(4j*PHI)*expm_multiply(-magnetic,v,traceA=0)
    return np.exp(-4j*PHI)*expm_multiply(magnetic,D*v,traceA=0)


def main():
    certpath=OUT/'STATIC_WILSON_CAR_CERTIFICATE.json';c=json.loads(certpath.read_text())
    data=OUT/'STATIC_WILSON_CAR_MODULES_AND_KERNELS.npz'
    assert sha(data)==c['data_sha256']
    primitive_path=ROOT/'reproducibility/results/nonabelian_common_car/NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json'
    primitive=json.loads(primitive_path.read_text())
    ground=primitive['cutoff_results'][-1]['ground_certificates']
    w=[F(v) for v in ground['SU2']['rational_witness_components']]
    N=sum(v*v for v in w)
    hw=[F(4*s*(s+2)+32)*v-16*((w[s-1] if s else 0)+(w[s+1] if s+1<len(w) else 0))
        for s,v in enumerate(w)]
    mu=sum(v*a for v,a in zip(w,hw))/N
    residual=(sum((a-mu*v)**2 for v,a in zip(w,hw))+(16*w[-1])**2)/N
    assert mu==F(ground['SU2']['rayleigh_rational'])
    assert residual==F(ground['SU2']['infinite_residual_squared_rational'])
    residual_bound=F(ground['SU2']['infinite_residual_upper_rational'])
    state_bound=F(ground['SU2']['state_trace_distance_upper_rational'])
    assert residual_bound**2>=residual and state_bound>=residual_bound/(40-mu)
    core=c['diagnostics']['weak_seed_core_radius']
    discarded=sum(v*v for v in w[core+1:])/N
    assert discarded==F(c['diagnostics']['weak_seed_discarded_norm_squared_rational'])
    core_bound=F(c['diagnostics']['weak_seed_trace_distance_upper_rational'])
    assert core_bound**2>=discarded
    rY=(len(ground['U1']['rational_witness_components'])-1)//2
    dY=504-2*rY
    ytail=min(F(2),2*F(33,5*dY)**dY)
    colour_tail=F(primitive['analytic_polynomial_tail_bounds_rational']['SU3'])
    colour_return=24*colour_tail*(1+colour_tail)**11
    Dy,Dw,Dc=[F(ground[g]['state_trace_distance_upper_rational']) for g in ('U1','SU2','SU3')]
    Dw+=core_bound
    for row in c['four_representation_boxes']:
        d=2*(row['twice_spin_box']+1-core)
        ret=min(F(2),2*F(264,5*d)**d) if F(5*d,96)>1 else F(2)
        assert F(row['closed_return_error_bound_rational'])>=ret
        assert F(row['physical_kernel_error_bound_rational'])>=min(F(2),2*(Dy+Dw)+2*(ret+ytail))
        assert F(row['comparison_moment_error_bound_rational'])>=min(F(2),4*(Dy+Dw+Dc)+4*(ret+ytail+colour_return))
    results=[]
    with np.load(data,allow_pickle=False) as z:
        for i,row in enumerate(c['module_checks']):
            a,b,d,e=map(lambda x:float(F(x)),row['quaternion_rational'])
            U=np.array([[a+1j*b,d+1j*e],[-d+1j*e,a-1j*b]])
            theta=row['angle_multiplier_pi_over_63']*PHI
            M=np.zeros((9,9),complex)
            for family in range(3):M[2*family:2*family+2,2*family:2*family+2]=np.exp(-3j*theta)*U
            M[6:,6:]=np.exp(-6j*theta)*np.eye(3)
            dense=exterior_power(M)
            exterior_error=float(np.linalg.norm(dense-csr(z,f'configuration{i}_fock_unitary').toarray()))
            module_error=float(sparse.linalg.norm(direct_module(U,theta)-csr(z,f'configuration{i}_module')))
            assert exterior_error<5e-13 and module_error<5e-13
            results.append({'configuration':i,'exterior_power_diagnostic':exterior_error,'direct_Wilson_product_diagnostic':module_error})
        blocks={};quadrature=[]
        seed=np.zeros(25,complex)
        radius=c['diagnostics']['weak_seed_core_radius']
        weak=np.array([float(F(v)) for v in ground['SU2']['rational_witness_components'][:radius+1]])
        weak/=np.linalg.norm(weak)
        seed[:len(weak)]=weak
        for h in range(4):
            labels,M,T,U,gram,source=quadrature_block(h)
            recorded_labels=z[f'weak_J{h}_labels']
            idx={tuple(map(int,v)):i for i,v in enumerate(recorded_labels)}
            selected=[idx[v] for v in labels]
            recorded=sparse.csr_matrix((z[f'weak_J{h}_magnetic_data'],z[f'weak_J{h}_magnetic_indices'],
              z[f'weak_J{h}_magnetic_indptr']),shape=(len(recorded_labels),len(recorded_labels)))
            error=float(np.linalg.norm(M-recorded[selected,:][:,selected].toarray(),2))
            assert error<4e-12
            blocks[h]=(T,U)
            quadrature.append({'twice_J':h,'Gram_diagnostic':gram,'source_diagnostic':source,'magnetic_diagnostic':error})
        U0=blocks[0][1]
        ns=[np.linalg.matrix_power(U0,n)@seed for n in range(13)]
        r0=np.array([np.vdot(seed,v) for v in ns])
        rcharged={};wk={}
        for h,(T,U) in blocks.items():
            initial=T@seed
            rcharged[h]=np.array([np.vdot(initial,np.linalg.matrix_power(U,n)@initial) for n in range(13)])
            fields=np.array([np.linalg.matrix_power(U.conj().T,n)@T@ns[n] for n in range(7)])
            wk[h]=fields.conj()@fields.T
        y=np.zeros(504,complex);small=np.array([float(F(v)) for v in ground['U1']['rational_witness_components']]);small/=np.linalg.norm(small)
        radius=(len(small)-1)//2;y[252-radius:253+radius]=small
        yn=[y]
        for _ in range(12):yn.append(y_step(0,yn[-1]))
        y0=np.array([np.vdot(y,v) for v in yn]);yc={};yk={}
        for Q in range(-36,1,3):
            vectors=[y]
            for _ in range(12):vectors.append(y_step(Q,vectors[-1]))
            yc[Q]=np.array([np.vdot(y,v) for v in vectors])
            fields=[]
            for n in range(7):
                v=yn[n]
                for _ in range(n):v=y_step(Q,v,True)
                fields.append(v)
            fields=np.array(fields);yk[Q]=fields.conj()@fields.T
        moment_error=kernel_error=0.
        spin_weights={n:tensor_irreps(n) for n in range(4)}
        for mask in range(64):
            nl=(mask&7).bit_count();nr=(mask>>3).bit_count();Q=-3*nl-6*nr
            weighted=sum(weight*rcharged[h] for h,weight in spin_weights[nl].items())
            C=y0.conj()*yc[Q]*r0.conj()*weighted*abs(z['neutral_colour_returns'])**2
            K=yk[Q]*sum(weight*wk[h] for h,weight in spin_weights[nl].items())
            moment_error=max(moment_error,float(np.max(abs(C-z['R24_comparison_moment_diagonal'][:,mask]))))
            kernel_error=max(kernel_error,float(np.max(abs(K-z['R24_physical_kernels'][mask]))))
        assert moment_error<2e-10 and kernel_error<2e-10
        # Direct orthogonal complement in the charged harmonic basis.
        T=blocks[1][0];labels=[(s,t) for s in range(25) for t in range(abs(s-1),s+2,2)]
        initial=T@seed
        electric=np.array([3*s*(s+2)+t*(t+2) for s,t in labels])
        evolved=np.exp(-1j*PHI*electric)*initial
        complement=evolved-T@(T.conj().T@evolved)
        squared=float(np.vdot(complement,complement).real)
        lo,hi=map(F,c['leakage']['primitive_ground_squared_leakage_interval_rational'])
        # Quadrature/diagonalization roundoff is assessed separately from the rational enclosure.
        assert float(lo)-2e-12<squared<float(hi)+2e-12
        witness=np.zeros_like(seed);witness[1]=1
        v=T@witness;e=np.exp(-1j*PHI*electric)*v
        small_squared=float(np.linalg.norm(e-T@(T.conj().T@e))**2)
        assert abs(small_squared-.75*math.sin(4*PHI)**2)<2e-13
    output={'status':'PASS_INDEPENDENT_STATIC_WILSON_CAR_VERIFICATION',
        'source_program_sha256':sha(Path(__file__)),'producer_certificate_sha256':sha(certpath),
        'data_sha256':sha(data),'module_checks':results,'angular_momentum_diagonalization_checks':quadrature,
        'tensor_product_spin_weights':spin_weights,
        'primitive_weak_Rayleigh_and_infinite_residual_exact':True,
        'core_state_and_representation_error_bounds_exact':True,
        'all_64_dense_comparison_moment_diagnostic':moment_error,
        'all_64_dense_physical_kernel_diagnostic':kernel_error,
        'direct_electric_complement_squared_diagnostic':squared,
        'j_half_complement_squared_diagnostic':small_squared,
        'floating_point_interval_certified':False,'full_C10_closure_inferred':False}
    (OUT/'INDEPENDENT_STATIC_WILSON_CAR_CERTIFICATE.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('module_checks','angular_momentum_diagonalization_checks')},indent=2))


if __name__=='__main__':main()
