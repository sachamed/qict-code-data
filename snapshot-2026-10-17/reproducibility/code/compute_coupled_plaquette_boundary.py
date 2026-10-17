#!/usr/bin/env python3
"""Authentic shared-link U(1) A7 state, charged update and coherent boundary.

Two adjacent stored plaquettes meet the original source edge at the origin.
The shared electric term is retained exactly. This is the Abelian component
on that isolated two-plaquette subcomplex, not the complete charged C10.
"""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math
import numpy as np
from scipy import sparse
from scipy.linalg import expm,eigh
from scipy.sparse.linalg import eigsh,expm_multiply,splu
from nonabelian_common_car_kato import sqrt_upper,polynomial_tail

ROOT=Path(__file__).resolve().parents[2]
PAYLOAD=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
OUT=ROOT/'reproducibility/results/coupled_boundary'
PHI=math.pi/63;DEPTH=12;ORDER=14


def operators(radius):
    labels=np.array([(r,s) for r in range(-radius,radius+1)
                     for s in range(-radius,radius+1)],np.int64)
    d=2*radius+1
    one=sparse.diags([np.ones(d-1),np.ones(d-1)],[-1,1],shape=(d,d),format='csr')
    fusion=sparse.kron(one,sparse.eye(d),format='csr')+sparse.kron(sparse.eye(d),one,format='csr')
    r,s=labels.T
    electric=8*r*r+8*s*s-4*r*s
    h=sparse.diags(electric+8,format='csr',dtype=np.float64)-2*fusion
    return labels,electric,fusion,h


def ground(radius):
    labels,electric,fusion,h=operators(radius)
    values,vectors=eigsh(h.astype(float),k=1,which='SA',tol=3e-15,
                         v0=np.exp(-np.sum(labels*labels,axis=1)/3))
    v=vectors[:,0]
    if v[np.argmax(abs(v))]<0:v=-v
    v/=np.linalg.norm(v)
    # Refine the eigenvector with a fixed numerical shift below the certified
    # ground enclosure from the initial witness; no physical coefficient changes.
    initial_residual=np.linalg.norm(h@v-values[0]*v)
    assert values[0]-initial_residual>6
    solve=splu((h-6*sparse.eye(h.shape[0],format='csr')).tocsc()).solve
    for _ in range(3):
        v=solve(v);v/=np.linalg.norm(v)
    w=[F(format(float(x),'.17g')) for x in v]
    lookup={tuple(x):i for i,x in enumerate(labels)}
    hv=[F(int(e+8))*x for e,x in zip(electric,w)];outside={}
    for i,((r,s),x) in enumerate(zip(labels,w)):
        for target in [(r-1,s),(r+1,s),(r,s-1),(r,s+1)]:
            if target in lookup:hv[lookup[target]]-=2*x
            else:outside[target]=outside.get(target,F(0))-2*x
    norm=sum(x*x for x in w)
    mu=sum(x*y for x,y in zip(w,hv))/norm
    residual2=(sum((y-mu*x)**2 for x,y in zip(w,hv))+
               sum(x*x for x in outside.values()))/norm
    epsilon=sqrt_upper(residual2);gap=F(8)-mu
    assert gap>0 and epsilon<gap
    record={'radius':radius,'rayleigh_rational':str(mu),'energy':float(mu),
      'residual_squared_rational':str(residual2),'residual_upper_rational':str(epsilon),
      'residual_upper':float(epsilon),'first_excited_lower':8,
      'gap_lower_rational':str(gap),'gap_lower':float(gap),
      'state_trace_distance_upper_rational':str(epsilon/gap),
      'state_trace_distance_upper':float(epsilon/gap),
      'omitted_neighbor_components':len(outside),
      'rational_witness_components':[str(x) for x in w]}
    return labels,v,record


def orbit(seed,electric,fusion):
    phase=np.exp(-1j*PHI*(electric%126))
    v=seed.copy();returns=[];normerror=0
    for n in range(DEPTH+1):
        returns.append(np.vdot(seed,v));normerror=max(normerror,abs(np.linalg.norm(v)-1))
        if n<DEPTH:v=np.exp(-8j*PHI)*expm_multiply(2j*PHI*fusion,phase*v,traceA=0)
    iv=seed.copy();inverse=[]
    for n in range(DEPTH+1):
        inverse.append(np.vdot(seed,iv))
        if n<DEPTH:iv=np.conj(phase)*np.exp(8j*PHI)*expm_multiply(-2j*PHI*fusion,iv,traceA=0)
    adj=float(np.max(abs(np.asarray(inverse)-np.conj(returns))))
    assert normerror<5e-12 and adj<5e-12
    return np.asarray(returns),{'orbit_norm_error':float(normerror),'adjoint_return_error':adj}


def sqrt_density(rho,rank=None):
    values,vectors=eigh((rho+rho.conj().T)/2)
    assert values.min()>-3e-13
    if rank is not None:
        values[:-rank]=0
    return (vectors*np.sqrt(np.maximum(values,0)))@vectors.conj().T


def boundary_check(radius,labels,v,edge):
    """Full HS, folded tau, and independent operator-Schmidt contraction."""
    _,electric,fusion,_=operators(radius)
    d=2*radius+1;dim=d*d
    magnetic=expm(-1j*PHI*(8*np.eye(dim)-2*fusion.toarray()))
    u0=magnetic@np.diag(np.exp(-1j*PHI*(electric%126)))
    r,s=labels.T
    uf={q:magnetic@np.diag(np.exp(-1j*PHI*((electric+4*q*(r-s)+2*q*q+q)%126))) for q in [-3,-6]}
    state=v.astype(complex)
    # A7 pure state and an actual finite C9 average, both on this carrier.
    evolved=[state]
    for _ in range(3):evolved.append(u0@evolved[-1])
    mixed=sum(np.outer(x,x.conj()) for x in evolved)/4
    cases={'A7_pure':np.outer(state,state.conj()),'C9_T4':mixed}
    errors=[];records=[]
    for name,rho in cases.items():
        root=sqrt_density(rho,1 if name=='A7_pure' else 4)
        x=root.copy()
        for q in [-3,-6]:
            x=root.copy();uq=uf[q];pn=np.eye(dim,dtype=complex);p0=pn.copy()
            for n in range(5):
                direct=np.vdot(root,x)
                a=pn@p0.conj().T
                tau=p0@root@p0.conj().T@root
                folded=np.trace(tau@a)
                naive_density_fold=np.trace(rho@a)
                # A= sum_l s_l F_l tensor E_l. E_l contains Vh's entries;
                # it is not conjugated or assigned an extra Kraus label.
                reshaped=a.reshape(d,d,d,d).transpose(0,2,1,3).reshape(d*d,d*d)
                left,sv,right=np.linalg.svd(reshaped,full_matrices=False)
                tensor=tau.reshape(d,d,d,d)
                # Physical Gauss boundary history: both exterior flux labels
                # are retained. A one-time density keeps only s=s' blocks.
                observable_blocks=a.reshape(d,d,d,d)
                sector_value=0j
                for ext in range(d):
                    for ext_prime in range(d):
                        sector_value+=np.trace(tensor[:,ext,:,ext_prime]@
                                               observable_blocks[:,ext_prime,:,ext])
                value=0j
                for j,sigma in enumerate(sv):
                    f=left[:,j].reshape(d,d);e=right[j].reshape(d,d)
                    theta=np.einsum('abcd,db->ac',tensor,e)
                    value+=sigma*np.trace(theta@f)
                err=max(abs(direct-folded),abs(direct-value),abs(direct-sector_value))
                errors.append(float(err))
                records.append({'state':name,'qY':q,'n':n,'HS_moment':[float(direct.real),float(direct.imag)],
                  'folded_Schmidt_error':float(err),'operator_Schmidt_singular_values':sv.tolist(),
                  'Gauss_boundary_history_block_error':float(abs(direct-sector_value)),
                  'transition_trace_norm':float(np.linalg.svd(tau,compute_uv=False).sum()),
                  'transition_hermiticity_defect':float(np.linalg.norm(tau-tau.conj().T)),
                  'ordinary_density_fold_difference':float(abs(direct-naive_density_fold))})
                x=uq@x@u0.conj().T;pn=uq@pn;p0=u0@p0
    assert max(errors)<3e-11
    amplitudes=state.reshape(d,d)
    # The actual core-plus-shared-link marginal has sector s and quantum
    # coordinate r: rho_s(r,r')=v(r,s)*conj(v(r',s)).
    blocks=np.einsum('rs,ts->srt',amplitudes,amplitudes.conj())
    off=blocks.copy()
    for b in off:np.fill_diagonal(b,0)
    block_trace=float(sum(np.trace(b).real for b in blocks))
    assert abs(block_trace-1)<2e-13 and np.linalg.norm(off)>.01
    # Dephase r as an added environment measurement; this is a different
    # reference channel, tested explicitly without calling it physical.
    rho=np.outer(state,state.conj()).reshape(d,d,d,d)
    deph=rho*np.eye(d)[:,None,:,None]
    deph=deph.reshape(dim,dim)
    roots=[np.outer(state,state.conj()),sqrt_density(deph,d)]
    changed=[]
    for q in [-3,-6]:
        values=[np.vdot(rt,uf[q]@rt@u0.conj().T) for rt in roots]
        changed.append({'qY':q,'coherent_C1':[float(values[0].real),float(values[0].imag)],
          'dephased_C1':[float(values[1].real),float(values[1].imag)],
          'absolute_change':float(abs(values[0]-values[1]))})
    unit=max(np.linalg.norm(a.conj().T@a-np.eye(dim)) for a in [u0,*uf.values()])
    # Two Gauss-invariant loop states have the same initial physical core
    # charge blocks. The crossing plaquette reveals their missing correlation.
    pair=[]
    for sign in [1,-1]:
        a=np.zeros((d,d),complex)
        a[radius,radius]=1/math.sqrt(2);a[radius+1,radius+1]=sign/math.sqrt(2)
        pair.append(a.reshape(dim))
    correlation=[]
    for n in range(4):
        b=[np.einsum('rs,ts->srt',a.reshape(d,d),a.reshape(d,d).conj()) for a in pair]
        distance=sum(np.linalg.svd(x-y,compute_uv=False).sum() for x,y in zip(*b))/2
        correlation.append({'n':n,'physical_core_trace_distance':float(distance)})
        pair=[u0@a for a in pair]
    assert correlation[0]['physical_core_trace_distance']==0 and correlation[1]['physical_core_trace_distance']>.01
    return blocks,{'small_regulator_dimension':dim,'unitarity_Frobenius_defect':float(unit),
                  'max_independent_HS_folded_boundary_error':max(errors),
                  'coherent_boundary_trace':block_trace,'coherent_boundary_offdiagonal_Frobenius_norm':float(np.linalg.norm(off)),
                  'dephasing_changes':changed,'moment_checks':records,
                  'correlated_state_same_boundary_witness':correlation,
                  'correlated_state_infinite_one_step_trace_distance':'sum_s |J_s(4*pi/63)*J_(s-1)(4*pi/63)|',
                  'correlated_state_infinite_one_step_distance_lower_rational':'9801/50000',
                  'correlated_state_analytic_bound':'199/1000 < x=4*pi/63 < 1/5; J0(x)>99/100 and J1(x)>99/1000 by alternating Bessel series; the s=0,1 terms give distance>9801/50000.',
                  'correlated_state_witness_scope':'Two admissible neutral Gauss loop states under the same authentic update; not two A7-selected ground states.',
                  'boundary_sector':'exterior plaquette flux s; r is a quantum multiplicity coordinate',
                  'boundary_history':'Keep both s and s_prime in tau_(s,s_prime); the loop-coordinate Schmidt cut is a separate algebraic factorization.',
                  'shared_source_link':int(edge)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(PAYLOAD,allow_pickle=False) as z:
        links=z['links'].astype(int);plaquettes=z['plaquettes'].astype(int)
        faces=z['plaquette_edge_indices'].astype(int);signs=z['plaquette_edge_sign'].astype(int)
    source_edge=int(np.flatnonzero(np.all(links==[0,0,0,0],axis=1))[0])
    p=int(np.flatnonzero(np.all(plaquettes==[0,1,0,0,0],axis=1))[0])
    candidates=[i for i,a in enumerate(faces) if source_edge in a and i!=p and len(set(a)&set(faces[p]))==1]
    q=min(candidates)
    orientations=[signs[p].copy(),signs[q].copy()]
    orientations[0]*=orientations[0][list(faces[p]).index(source_edge)]
    orientations[1]*=-orientations[1][list(faces[q]).index(source_edge)]
    union=sorted(set(faces[p])|set(faces[q]));B=np.zeros((len(union),2),int)
    for j,face in enumerate([p,q]):
        for e,c in zip(faces[face],orientations[j]):B[union.index(int(e)),j]=c
    assert np.array_equal(B.T@B,[[4,-1],[-1,4]]) and np.all(B.sum(axis=0)==0)
    arrays={'actual_plaquette_ids':np.array([p,q]),'actual_link_indices':np.array(union),
            'integer_cycle_matrix':B,'source_positive_edge':np.array(source_edge)}
    records=[];previous=None;small=None
    tail=polynomial_tail(F(2,5),ORDER)
    for level,radius in enumerate([4,6,8,10]):
        glabels,v,cert=ground(radius)
        pad=radius+DEPTH*ORDER+2
        labels,electric,fusion,h=operators(pad)
        d=2*pad+1;seed=np.zeros(len(labels),complex)
        seed[(glabels[:,0]+pad)*d+glabels[:,1]+pad]=v
        a0,diagnostics=orbit(seed,electric,fusion)
        cs=[];returns=[]
        r,s=labels.T
        for charge in [-3,-6]:
            b,diag=orbit(seed,electric+4*charge*(r-s)+2*charge*charge+charge,fusion)
            returns.append(b);cs.append(a0.conj()*b);diagnostics[str(charge)]=diag
        c=np.zeros((DEPTH+1,6,6),complex)
        for n in range(DEPTH+1):c[n]=np.diag([cs[0][n]]*3+[cs[1][n]]*3)
        toeplitz=np.block([[c[j-i] if j>=i else c[i-j].conj().T
                           for j in range(DEPTH+1)] for i in range(DEPTH+1)])
        minimum=float(np.linalg.eigvalsh(toeplitz).min())
        assert minimum>-1e-11 and np.linalg.norm(c[0]-np.eye(6))<2e-11
        error=4*F(cert['state_trace_distance_upper_rational'])+4*DEPTH*tail
        change=None if previous is None else float(np.max(abs(c-previous)))
        records.append({'level':level,'ground':cert,'dynamical_flux_radius':pad,
          'dimension':len(labels),'analytic_state_and_padding_error_rational':str(error),
          'analytic_state_and_padding_error_upper':float(error),
          'moment_successive_max_change':change,'Toeplitz_minimum_diagnostic':minimum,
          'floating_point_diagnostics':diagnostics})
        arrays[f'L{level}_Cn6']=c;arrays[f'L{level}_ground_labels']=glabels
        arrays[f'L{level}_ground_vector']=v;arrays[f'L{level}_neutral_return']=a0
        arrays[f'L{level}_charged_returns']=np.asarray(returns)
        previous=c
        if level==0:small=(glabels,v)
        print(json.dumps({'radius':radius,'E0':cert['energy'],'gap_lower':cert['gap_lower'],
                          'moment_change':change,'analytic_error':float(error)}),flush=True)
    blocks,independent=boundary_check(4,*small,source_edge)
    arrays['small_coherent_boundary_charge_blocks']=blocks
    npz=OUT/'COUPLED_PLAQUETTE_BOUNDARY_MOMENTS.npz';np.savez_compressed(npz,**arrays)
    result={'status':'AUTHENTIC_SHARED_LINK_ABELIAN_GAUGE_BOUNDARY_AND_COMMON_GNS_CONTRACTION',
      'payload_sha256':hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),
      'actual_plaquette_ids':[p,q],'actual_source_edge':source_edge,
      'source_position':[0,0,0],'compensator_position':[1,0,0],
      'neutral_electric_generator':'8*r^2+8*s^2-4*r*s',
      'charged_electric_generator':'8*r^2+8*s^2-4*r*s+4*qY*(r-s)+2*qY^2+qY',
      'magnetic_generator':'8-2*(S_r+S_r^dagger+S_s+S_s^dagger)',
      'selected_state':'A7 coupled gauge ground on the isolated two-plaquette component',
      'source_Gram':'I6 on empty CAR reference, affine-flux Wilson dressing and dual endpoint',
      'source_scope':'U1 components of the six leptonic representation labels; weak/color and BCC streams are not included in this subcomplex calculation.',
      'GNS_transfer':'conjugate(U0) tensor Uf on HS(H0,Hf)',
      'general_boundary_transition':'tau_n=U0^n sqrt(rho) U0^-n sqrt(rho)',
      'folded_observable':'A_ab(n)=O_a^dagger Uf^n O_b U0^-n; C_ab=Tr(tau_n*A_ab)',
      'cutoffs':records,'independent_dense_and_Schmidt_checks':independent,
      'floating_point_interval_certified':False,
      'full_C10_contract_completed':False,'spatial_buffer_limit_certified':False,
      'data_sha256':hashlib.sha256(npz.read_bytes()).hexdigest()}
    (OUT/'COUPLED_BOUNDARY_GNS_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'max_boundary_check_error':independent['max_independent_HS_folded_boundary_error'],
                      'boundary_coherence_norm':independent['coherent_boundary_offdiagonal_Frobenius_norm'],
                      'full_C10':False},indent=2))


if __name__=='__main__':main()
