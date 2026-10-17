#!/usr/bin/env python3
"""Coherent source/complement memory on the archived static plaquette.

All weak blocks come from the distributed symbolic recoupling data. The
central Fourier calculation is a magnetic-operator symbol, not a C10 state
contraction. Kernel comparisons use the existing integer-arithmetic witness.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib, io, json, math, platform
import numpy as np
import scipy
from scipy import sparse
from scipy.linalg import expm

from prepare_inputs import read_reference
from certify_static_kernel_arithmetic import SCALE, pi_bounds, phases, normalized_seed, weak_update, gram

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'reproducibility/results/static_wilson_car'
OUT=ROOT/'reproducibility/results/coherent_source_memory'
DEPTH=12
PHI=math.pi/63


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def outward(x,up=True,denominator=10**16):
    n=x*denominator
    a=n.numerator//n.denominator
    if up and n.denominator!=1:a+=1
    return F(a,denominator)


def norm(a):return float(np.linalg.norm(a,2)) if a.size else 0.0


def relative_residual(a,b):return float(np.linalg.norm(a-b,ord='fro'))


def source_frame(labels,T):
    """Complete the source inside each fixed outer-spin block."""
    q=[]
    for s in range(T.shape[1]):
        rows=np.flatnonzero(labels[:,0]==s)
        v=T[rows,s]
        assert abs(np.linalg.norm(v)-1)<2e-14
        if len(rows)>1:
            local,_=np.linalg.qr(v[:,None],mode='complete')
            for k in range(1,len(rows)):
                column=np.zeros(T.shape[0]);column[rows]=local[:,k];q.append(column)
    Q=np.stack(q,axis=1) if q else np.zeros((T.shape[0],0))
    frame=np.concatenate([T,Q],axis=1)
    assert frame.shape[0]==frame.shape[1]
    assert relative_residual(frame.T@frame,np.eye(len(frame)))<1e-13
    return Q,frame


def unitary_blocks(z,h):
    labels=z[f'weak_J{h}_labels']
    d=len(labels);ns=int(labels[:,0].max())+1
    M=sparse.csr_matrix((z[f'weak_J{h}_magnetic_data'],
                        z[f'weak_J{h}_magnetic_indices'],z[f'weak_J{h}_magnetic_indptr']),shape=(d,d)).toarray()
    T=sparse.csr_matrix((z[f'weak_J{h}_source_data'],
                        z[f'weak_J{h}_source_indices'],z[f'weak_J{h}_source_indptr']),shape=(d,ns)).toarray()
    E=z[f'weak_J{h}_electric']
    Q,frame=source_frame(labels,T)
    electric=np.diag(np.exp(-1j*PHI*(E%126)))
    magnetic=np.exp(-32j*PHI)*expm(16j*PHI*M)
    U=magnetic@electric
    A=T.T@U@T;B=T.T@U@Q;C=Q.T@U@T;D=Q.T@U@Q
    return labels,T,Q,frame,U,magnetic,electric,A,B,C,D


def source_returns(A,B,C,D,depth):
    memory=[];power=np.eye(D.shape[0],dtype=complex)
    for _ in range(depth):
        memory.append(B@power@C);power=D@power
    result=[np.eye(len(A),dtype=complex)]
    for n in range(depth):
        following=A@result[n]
        for j in range(n):following+=memory[n-j-1]@result[j]
        result.append(following)
    return np.asarray(result),np.asarray(memory)


def kernels(neutral,returns):
    k=np.zeros((7,7),complex)
    for t in range(7):
        for s in range(7):
            transfer=returns[t-s] if t>=s else returns[s-t].conj().T
            k[t,s]=np.vdot(neutral[t],transfer@neutral[s])
    return k


def y_kernel(Q,states):
    r=np.arange(504)-252
    cost=(8*r*r+4*Q*r+2*Q*Q+Q)%126
    electric=np.exp(1j*PHI*cost)
    magnetic=np.exp(1j*PHI*(4-4*np.cos(2*np.pi*np.arange(504)/504)))
    fields=[]
    for t,state in enumerate(states[:7]):
        v=state.copy()
        for _ in range(t):v=electric*np.fft.ifft(magnetic*np.fft.fft(v))
        fields.append(v)
    fields=np.asarray(fields)
    return fields.conj()@fields.T


def central_symbol(arrays):
    costs=np.array([0,120,60,72,60,120],np.int64)
    index=np.arange(6)
    fourier=np.exp(-2j*np.pi*np.outer(index,index)/6)/math.sqrt(6)
    H=fourier@np.diag(costs)@fourier.conj().T
    V=fourier@np.diag(np.exp(-1j*PHI*(costs%126)))@fourier.conj().T
    shift=np.roll(np.eye(6),1,axis=0)
    polynomial=72*np.eye(6)-2*(shift+shift.T)-32*np.linalg.matrix_power(shift,3)-18*(shift@shift+shift.T@shift.T)
    assert relative_residual(H,polynomial)<2e-12
    assert relative_residual(V.conj().T@V,np.eye(6))<2e-14
    table,phase_error=phases(pi_bounds())
    real=imag=F(0)
    for cost in costs:
        a,b=table[(int(cost)+63)%126-63]
        real+=F(a,6*SCALE);imag-=F(b,6*SCALE)
    centre=real*real+imag*imag
    error=2*phase_error+3*phase_error**2
    lower=outward(1-centre-error,False);upper=outward(1-centre+error)
    arrays['central_symbol_electric_costs']=costs
    arrays['central_symbol_magnetic_hamiltonian']=H
    arrays['central_symbol_magnetic_unitary']=V
    return {'carrier':'Six-point central orbit symbol of the product-cover magnetic multiplier.',
        'central_costs':costs.tolist(),'central_average_cost_exact':72,
        'sector_transition_charge_residues':[1,5,3,2,4],
        'trivial_sector_loss_squared_interval_rational':[str(lower),str(upper)],
        'symbol_unitarity_diagnostic':relative_residual(V.conj().T@V,np.eye(6)),
        'symbol_polynomial_diagnostic':relative_residual(H,polynomial),
        'quotient_only_sector_invariant':False,
        'physical_C10_state_kernel':False}


def face_cycles():
    content=json.loads((ROOT/'reproducibility/verification/SCIENTIFIC_CONTENT.json').read_text())
    name='reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    ref=content['logical_scientific_files'][name]
    with np.load(io.BytesIO(read_reference(ref)),allow_pickle=False) as z:
        links=z['links'].astype(int);edges=z['plaquette_edge_indices'];sign=z['plaquette_edge_sign'].astype(int)
        starts=links[:,1:];ends=starts.copy();ends[np.arange(len(links)),links[:,0]]+=1
        assert edges.shape==sign.shape==(18252,4)
        # Character-coded endpoints expose the exact signed incidence identity.
        vertices,inv=np.unique(np.concatenate([starts,ends]),axis=0,return_inverse=True)
        a,b=inv[:len(links)],inv[len(links):]
        for face in range(len(edges)):
            boundary={}
            for edge,orientation in zip(edges[face],sign[face]):
                for vertex,value in ((a[edge],-orientation),(b[edge],orientation)):
                    boundary[int(vertex)]=boundary.get(int(vertex),0)+int(value)
            assert all(x==0 for x in boundary.values())
    return {'payload_sha256':ref['sha256'],'faces_checked_integer_incidence':18252,
        'links':len(links),'vertices':len(vertices),'cycle_rank_connected':len(links)-len(vertices)+1,
        'central_character_increments':[1,-1,3,2,-2],
        'Gauss_divergence_change_per_face_exact':0,
        'full_routed_matter_gate_checked':False}


def primitive_stationarity(z,producer,intervals):
    """Enclose the weak primitive state's one-step trace-distance squared."""
    primitive=ROOT/'reproducibility/results/nonabelian_common_car/NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json'
    old=json.loads(primitive.read_text())
    values=old['cutoff_results'][-1]['ground_certificates']['SU2']['rational_witness_components']
    values=values[:producer['diagnostics']['weak_seed_core_radius']+1]
    pi=pi_bounds();table,phase_error=phases(pi)
    U,_,_=weak_update(z,0,pi,table,phase_error)
    seed=normalized_seed(values,61)
    advanced=U.apply(seed)
    real,imag,radius=gram([seed,advanced])[0][1]
    radius+=F(intervals['analytic_state_and_representation_bound_rational'])
    error=2*radius+3*radius**2
    centre=1-real*real-imag*imag
    lower,upper=outward(centre-error,False),outward(centre+error)
    assert lower>0
    return {'weak_primitive_trace_distance_squared_interval_rational':[str(lower),str(upper)],
        'exact_identity':'D(|Omega><Omega|, U|Omega><Omega|U*)^2 = 1 - |<Omega,U Omega>|^2',
        'primitive_weak_density_stationary_under_frozen_update':False,
        'arithmetic':'Integer propagation with rational state and representation majorants.',
        'scope':'The isolated weak primitive state, not the full interacting C10 vacuum.'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=SOURCE/'STATIC_WILSON_CAR_MODULES_AND_KERNELS.npz'
    producer_path=SOURCE/'STATIC_WILSON_CAR_CERTIFICATE.json'
    producer=json.loads(producer_path.read_text());assert sha(data)==producer['data_sha256']
    interval_path=SOURCE/'STATIC_PHYSICAL_KERNEL_ARITHMETIC_CERTIFICATE.json'
    intervals=json.loads(interval_path.read_text())
    centres_path=SOURCE/'STATIC_PHYSICAL_KERNEL_INTERVAL_CENTRES.npz'
    assert sha(centres_path)==intervals['interval_centre_data_sha256']
    arrays={};records=[];weak={};maximum_memory=maximum_defect=0.
    with np.load(data,allow_pickle=False) as z:
        stationarity=primitive_stationarity(z,producer,intervals)
        neutral=z['neutral_weak_states'][:7]
        for h in range(4):
            labels,T,Q,frame,U,M,E,A,B,C,D=unitary_blocks(z,h)
            R,memory=source_returns(A,B,C,D,DEPTH)
            direct=[];power=np.eye(len(U),dtype=complex)
            for n in range(DEPTH+1):direct.append(T.T@power@T);power=U@power
            direct=np.asarray(direct)
            recurrence_error=float(np.max(abs(direct-R)))
            defect=relative_residual(np.eye(len(A))-A.conj().T@A,C.conj().T@C)
            magnetic_defect=relative_residual(Q.T@M@T,np.zeros((Q.shape[1],T.shape[1])))
            Bm=T.T@M@Q;Cm=Q.T@M@T;Dm=Q.T@M@Q;Am=T.T@M@T
            Ae=T.T@E@T;Be=T.T@E@Q;Ce=Q.T@E@T;De=Q.T@E@Q
            composition=max(relative_residual(A,Am@Ae+Bm@Ce),
                relative_residual(B,Am@Be+Bm@De),relative_residual(C,Cm@Ae+Dm@Ce),
                relative_residual(D,Cm@Be+Dm@De))
            maximum_memory=max(maximum_memory,recurrence_error);maximum_defect=max(maximum_defect,defect)
            assert recurrence_error<3e-13 and defect<2e-13 and magnetic_defect<2e-13 and composition<2e-13
            weak[h]=kernels(neutral,R)
            resolver=[]
            for point in (2+0j,1.5+.25j):
                resolvent=T.T@np.linalg.solve(point*np.eye(len(U))-U,T)
                sigma=B@np.linalg.solve(point*np.eye(len(D))-D,C) if len(D) else np.zeros_like(A)
                effective=np.linalg.solve(point*np.eye(len(A))-A-sigma,np.eye(len(A)))
                sigma_tail=sum((point**(-k-1))*memory[k] for k in range(DEPTH))
                delta=abs(point)**(-DEPTH)/(abs(point)-1)
                assert norm(sigma-sigma_tail)<=delta+1e-12
                resolver.append({'z':[point.real,point.imag],
                    'Feshbach_equality_diagnostic':relative_residual(resolvent,effective),
                    'memory_tail_operator_norm_diagnostic':norm(sigma-sigma_tail),
                    'analytic_memory_tail_bound':delta})
            compressed=neutral[0].copy();coherent=[];markov=[]
            for n in range(DEPTH+1):
                coherent.append(np.vdot(neutral[0],R[n]@neutral[0]))
                markov.append(np.vdot(neutral[0],compressed));compressed=A@compressed
            seed_loss=float(np.linalg.norm(C@neutral[0])**2)
            arrays[f'J{h}_source_frame']=frame
            arrays[f'J{h}_unitary']=U
            arrays[f'J{h}_source_returns']=R
            arrays[f'J{h}_memory_kernels']=memory
            arrays[f'J{h}_coherent_seed_returns']=np.asarray(coherent)
            arrays[f'J{h}_compressed_seed_returns']=np.asarray(markov)
            records.append({'twice_spin':h,'charged_dimension':len(U),'source_dimension':len(A),
                'complement_dimension':len(D),'recurrence_maximum_entry_diagnostic':recurrence_error,
                'unitarity_frobenius_diagnostic':relative_residual(U.conj().T@U,np.eye(len(U))),
                'source_frame_unitarity_diagnostic':relative_residual(frame.T@frame,np.eye(len(U))),
                'positive_defect_identity_diagnostic':defect,'magnetic_source_complement_diagnostic':magnetic_defect,
                'ordered_gate_block_composition_diagnostic':composition,
                'seed_squared_complement_diagnostic':seed_loss,
                'coherent_vs_repeated_compression_maximum_seed_return_difference':float(np.max(abs(np.asarray(coherent)-markov))),
                'resolvents':resolver})
            print('coherent memory block',h,'dimensions',len(A),len(D),flush=True)
        Y={q:y_kernel(q,z['neutral_U1_states']) for q in range(-36,1,3)}
        weights={0:{0:1},1:{1:1},2:{0:.25,2:.75},3:{1:.5,3:.5}}
        result=[]
        for mask in range(64):
            nl=(mask&7).bit_count();nr=(mask>>3).bit_count();q=-3*nl-6*nr
            result.append(Y[q]*sum(w*weak[h] for h,w in weights[nl].items()))
        arrays['physical_kernels_coherent_memory']=np.asarray(result)
        arrays['physical_kernels_preceding_difference']=np.asarray(result)-z['R60_physical_kernels']
    # Enclose the actual reported values against the independent integer centres.
    radius=F(intervals['maximum_finite_kernel_arithmetic_radius_rational'])+F(intervals['analytic_state_and_representation_bound_rational'])
    maximum=F(0)
    with np.load(centres_path,allow_pickle=False) as z:
        denominator=int(str(z['common_denominator']))
        centres=z['gaussian_integer_centres']
        for mask in range(64):
            nl=(mask&7).bit_count();nr=(mask>>3).bit_count()
            for t in range(7):
                for s in range(7):
                    real,imag=(F(int(x),denominator) for x in centres[4*nl+nr,t,s])
                    v=arrays['physical_kernels_coherent_memory'][mask,t,s]
                    error=abs(F(float(v.real))-real)+abs(F(float(v.imag))-imag)+2*radius
                    maximum=max(maximum,error)
    # The uniform operator norm is the half-angle cosine.
    lo,hi=pi_bounds();x=(lo+hi)/252
    centre=sum(((-1)**k*x**(2*k)/math.factorial(2*k) for k in range(32)),F(0))
    err=(hi-lo)/252+x**64/math.factorial(64)
    uniform_interval=[outward(centre-err,False),outward(centre+err)]
    modular=central_symbol(arrays)
    cycles=face_cycles()
    output=OUT/'COHERENT_SOURCE_MEMORY_DATA.npz';np.savez_compressed(output,**arrays)
    certificate={'status':'PASS_COHERENT_SOURCE_MEMORY_AND_CENTRAL_LIFT',
        'source_program_sha256':sha(Path(__file__)),'data_sha256':sha(output),
        'preceding_source_certificate_sha256':sha(producer_path),'preceding_source_data_sha256':sha(data),
        'preceding_kernel_arithmetic_certificate_sha256':sha(interval_path),
        'interval_centre_data_sha256':sha(centres_path),
        'memory_depth':DEPTH,'static_occupation_sectors':64,'physical_time_range':[0,6],
        'weak_memory_blocks':records,'maximum_return_recurrence_entry_diagnostic':maximum_memory,
        'maximum_positive_defect_identity_diagnostic':maximum_defect,
        'reported_memory_kernel_entry_error_upper_rational':str(outward(maximum)),
        'reported_memory_kernel_seven_time_operator_error_upper_rational':str(outward(7*maximum)),
        'kernel_error_method':'Exact binary-rational comparison to independently enclosed Gaussian-integer kernel centres.',
        'one_left_uniform_complement_operator_norm_exact':'cos(pi/126)',
        'one_left_uniform_complement_operator_norm_interval_rational':[str(v) for v in uniform_interval],
        'primitive_squared_complement_interval_rational':producer['leakage']['primitive_ground_squared_leakage_interval_rational'],
        'primitive_stationarity':stationarity,
        'central_lift':modular,'archived_face_cycle_checks':cycles,
        'new_physical_parameters':0,'full_moving_C10_contract_completed':False,
        'spectral_and_resolvent_arithmetic':'Floating-point diagnostics and analytic tails; physical kernel output has a separate deterministic comparison enclosure.',
        'environment':{'Python':platform.python_version(),'NumPy':np.__version__,'SciPy':scipy.__version__}}
    (OUT/'COHERENT_SOURCE_MEMORY_CERTIFICATE.json').write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in certificate.items() if k in ('status','reported_memory_kernel_entry_error_upper_rational','one_left_uniform_complement_operator_norm_interval_rational','central_lift')},indent=2))


if __name__=='__main__':main()
