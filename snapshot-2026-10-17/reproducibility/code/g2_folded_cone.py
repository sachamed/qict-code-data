#!/usr/bin/env python3
"""Exact folded-Heisenberg rank-six moments for a *supplied* local unitary circuit.

This is an alternative to computing U_cone^n V by six full-state propagations.
It contracts O_i^dagger U^n O_j U^{-n} against reduced selected-state
marginals.  It does not infer the charged QICT gates or Kato marginals from
classical C10 probabilities, and never labels test-circuit moments physical.

For the normalized selected eigenstate U|Omega> = lambda|Omega>,
V[:,j]=O_j|Omega>, C_n[i,j]=lambda**n <Omega|O_i^dagger
 (U^n O_j U^{-n})|Omega>.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import hashlib, json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class LocalGate:
    sites: tuple[int, ...]
    matrix: np.ndarray

    def validate(self, nsites: int, d: int=2, tol: float=5e-13):
        if len(set(self.sites)) != len(self.sites) or not self.sites:
            raise ValueError('gate sites must be nonempty and unique')
        if any(not 0 <= a < nsites for a in self.sites):
            raise ValueError('gate site out of range')
        U=np.asarray(self.matrix, dtype=complex)
        dim=d**len(self.sites)
        if U.shape != (dim,dim): raise ValueError('gate shape inconsistent with support')
        if np.linalg.norm(U.conj().T@U-np.eye(dim),2)>tol:
            raise ValueError('nonunitary local gate: physical U_cone needs coherent amplitudes')


def digits(n: int, count: int, base: int)->tuple[int,...]:
    out=[0]*count
    for j in range(count-1,-1,-1): out[j]=n%base; n//=base
    return tuple(out)


def embed(A: np.ndarray, target: tuple[int,...], carrier: tuple[int,...], d:int=2)->np.ndarray:
    """Embed a local operator, preserving the specified *ordered* target axes.

    Small-cone exact implementation; not a large-volume tensor network engine.
    """
    if len(set(carrier))!=len(carrier) or len(set(target))!=len(target) or not set(target)<=set(carrier):
        raise ValueError('invalid carrier or target')
    A=np.asarray(A,complex)
    if A.shape!=(d**len(target),d**len(target)):
        raise ValueError('operator dimension does not match its sites')
    dim=d**len(carrier); pos=[carrier.index(a) for a in target]
    excluded=[j for j in range(len(carrier)) if j not in pos]
    tt=np.zeros(dim,dtype=np.int64); rr=np.zeros(dim,dtype=np.int64)
    for i in range(dim):
        dig=digits(i,len(carrier),d)
        for k in pos: tt[i]=tt[i]*d+dig[k]
        for k in excluded: rr[i]=rr[i]*d+dig[k]
    return np.where(rr[:,None]==rr[None,:],A[tt[:,None],tt[None,:]],0.0)


def circuit_to_dense(nsites:int, gates:list[LocalGate], d:int=2)->np.ndarray:
    """Reference implementation, used only to independently check small systems."""
    allsites=tuple(range(nsites)); U=np.eye(d**nsites,dtype=complex)
    for g in gates: U=embed(g.matrix,g.sites,allsites,d)@U
    return U


def heisenberg_cone(source:np.ndarray,source_sites:tuple[int,...],gates:list[LocalGate],depth:int,
                     *, nsites:int, local_d:int=2, max_cone_sites:int=9):
    """Conjugate with the *actual* chronological circuit, removing disjoint gates.

    The operator support grows only upon intersection, never by an arbitrary
    shell envelope. Reordering independent sites is handled by embed().
    """
    if depth<0: raise ValueError('depth must be nonnegative')
    support=tuple(source_sites)
    O=np.asarray(source,complex).copy(); used=0; skipped=0; growth=[]
    for _ in range(depth):
        for g in gates:
            if not (set(g.sites)&set(support)):
                skipped+=1; continue
            new=tuple(sorted(set(support)|set(g.sites)))
            if len(new)>max_cone_sites:
                raise MemoryError(f'operator cone grows to {len(new)} sites: supply a coherent MPO/tensor-network backend')
            if new!=support:
                O=embed(O,support,new,local_d);support=new
            Q=embed(g.matrix,g.sites,support,local_d)
            O=Q@O@Q.conj().T
            used+=1
        growth.append(len(support))
    return O,support,{'used_gates':used,'skipped_gates':skipped,'sites_by_step':growth}


def folded_rank6_moments(nsites:int, gates:list[LocalGate], operators:list[tuple[np.ndarray,tuple[int,...]]],
                          density_oracle:Callable[[tuple[int,...]],np.ndarray]|None, nmax:int,
                          eigenvalue:complex|None, *, local_d:int=2, max_cone_sites:int=9,
                          transition_oracle:Callable[[int,tuple[int,...]],np.ndarray]|None=None):
    """Exact source-space C_n for exact gates, eigenstate phase and reduced rho.

    The benchmark source carrier is V_a=O_a Omega on the Hilbert space of U.
    For relative GNS columns O_a sqrt(rho), the supplied transition must
    instead be the reduction of U0^n sqrt(rho) U0^-n sqrt(rho), with charged
    Heisenberg transport on the same direct-sum sector carrier. The plain
    rank-one transition used by the nonstationary benchmark below computes
    the one-leg compression; it is not the common neutral/charged GNS result.

    The oracle returns physical joint *coherent* reduced density on the
    requested support. Sector mixtures specified only by populations are not
    sufficient. The source Gram is audited at n=0.
    """
    if len(operators)!=6: raise ValueError('exactly six source operators required')
    if transition_oracle is None and (eigenvalue is None or abs(abs(eigenvalue)-1)>1e-11):
        raise ValueError('stationary folded contraction requires a unit-modulus state eigenphase')
    for g in gates:g.validate(nsites,local_d)
    moments=np.zeros((nmax+1,6,6),complex); rows=[]
    for n in range(nmax+1):
        row={'n':n,'used_gates':0,'skipped_gates':0,'max_cone_sites':0}
        for j,(Oj,Sj) in enumerate(operators):
            evolved,support,stat=heisenberg_cone(Oj,Sj,gates,n,nsites=nsites,
                                    local_d=local_d,max_cone_sites=max_cone_sites)
            row['used_gates']+=stat['used_gates'];row['skipped_gates']+=stat['skipped_gates']
            for i,(Oi,Si) in enumerate(operators):
                carrier=tuple(sorted(set(support)|set(Si)))
                if len(carrier)>max_cone_sites:raise MemoryError('requested expectation exceeds certified cone capacity')
                row['max_cone_sites']=max(row['max_cone_sites'],len(carrier))
                evolved_carrier=embed(evolved,support,carrier,local_d)
                left=embed(Oi,Si,carrier,local_d).conj().T
                if transition_oracle is not None:
                    rho=np.asarray(transition_oracle(n,carrier),complex)
                    prefactor=1.0
                else:
                    rho=np.asarray(density_oracle(carrier),complex)
                    prefactor=eigenvalue**n
                D=local_d**len(carrier)
                if rho.shape!=(D,D):raise ValueError('Kato marginal/transition has wrong shape')
                if transition_oracle is not None:
                    if np.linalg.svd(rho,compute_uv=False).sum()>1+2e-10:
                        raise ValueError('coherent transition trace norm exceeds a normalized rank-one transition')
                else:
                    if abs(np.trace(rho)-1)>2e-11 or np.linalg.norm(rho-rho.conj().T)>2e-11:
                        raise ValueError('Kato marginal is not a normalized Hermitian density')
                    if np.linalg.eigvalsh((rho+rho.conj().T)/2).min() < -2e-11:
                        raise ValueError('Kato marginal fails positive-semidefinite check')
                # Transition matrix is generally NON-Hermitian; retain all phases.
                moments[n,i,j]=prefactor*np.trace(rho@(left@evolved_carrier))
        rows.append(row)
    if np.linalg.norm(moments[0]-np.eye(6),'fro')>2e-10:
        raise ValueError('Six source columns fail C0=I; this is not a normalized receiver injection')
    return moments,rows


def toeplitz(C:np.ndarray)->np.ndarray:
    n=len(C)-1; d=C.shape[1];B=np.empty(((n+1)*d,(n+1)*d),complex)
    for i in range(n+1):
        for j in range(n+1):
            k=j-i
            # The other convention yields the transpose conjugate block Toeplitz;
            # both are Gram-positive for a unitary compression.
            B[i*d:(i+1)*d,j*d:(j+1)*d]=C[k] if k>=0 else C[-k].conj().T
    return (B+B.conj().T)/2


def sine_taper(N:int)->np.ndarray:
    u=np.sqrt(2/(N+2))*np.sin(np.arange(1,N+2)*np.pi/(N+2))
    return np.array([np.vdot(u[n:],u[:N+1-n]).real for n in range(N+1)])


def resolvent_from_positive_moments(C:np.ndarray,z:complex)->dict:
    if abs(z)<=1:raise ValueError('positive exterior resolvent requires |z|>1')
    if C.shape[1:]!=(6,6):raise ValueError('expected six sources')
    N=len(C)-1;b=sine_taper(N)
    G=sum(b[n]*C[n]/z**(n+1) for n in range(N+1))
    sigma=z*np.eye(6)-C[1]-np.linalg.inv(G) if N>=1 else None
    r=abs(z); delta=1-np.cos(np.pi/(N+2))
    return {'G':G,'Sigma':sigma,'resolvent_inverse_norm':float(np.linalg.norm(np.linalg.inv(G),2)),
            'inverse_norm_bound':r+1,'selfenergy_bound':float(delta*(r+1)**3/(r-1)**3) if N>=1 else None,
            'N':N,'taper':b}


def operator_error_fro_bound(depth:int, eigenstate_residual:float, trace_distance_to_marginal:float,
                             sum_gate_two_norm_errors_in_cone:float,source_operator_norm_max:float=1.0)->float:
    """Conservative, triangle-inequality bound (with trace norm, not half-distance).

    For exact unitary gates of norm one and isometric local source operators:
    entrywise error <= opnorm**2 *(n*residual + trace_norm_error + 2*sum_gate_errors).
    Frobenius of a 6x6 matrix <=6 times the entrywise worst case.
    The gate errors must be certified operator 2-norm bounds, and the oracle
    error is ||rho_exact-rho_approx||_1, not half the trace distance.
    """
    vals=(eigenstate_residual,trace_distance_to_marginal,sum_gate_two_norm_errors_in_cone)
    if any(v<0 or not np.isfinite(v) for v in vals):raise ValueError('nonnegative certified bounds required')
    return 6*source_operator_norm_max**2*(depth*eigenstate_residual+trace_distance_to_marginal+
                                           2*sum_gate_two_norm_errors_in_cone)


def exchange(theta:float,phase:float=0.0)->np.ndarray:
    """Excitation-preserving two-site gate; used ONLY as an algorithmic benchmark."""
    c=np.cos(theta);s=np.sin(theta)
    G=np.eye(4,dtype=complex)
    G[1,1]=G[2,2]=c
    G[1,2]=1j*np.exp(1j*phase)*s
    G[2,1]=1j*np.exp(-1j*phase)*s
    return G


def entangled_state_control_test(nsites:int=6,nmax:int=3)->dict:
    """Independent CAR-parity-even gate example with a non-product Kato marginal."""
    ops=[(np.array([[0,0],[1,0]],complex),(j,)) for j in range(6)]
    base=[LocalGate((a,a+1),exchange(.17+.061*a,.09*a)) for a in (0,2,4,1,3)]
    t=.37;c=np.cos(t);s=np.sin(t)
    # Even fermion-parity Bogoliubov/pair gate: |00> <-> |11>.
    W=np.eye(4,dtype=complex);W[0,0]=W[3,3]=c;W[0,3]=W[3,0]=1j*s
    entangler=LocalGate((0,1),W)
    gates=[LocalGate((0,1),W.conj().T)]+base+[entangler]
    carrier=tuple(range(nsites));D=2**nsites
    Wfull=embed(W,(0,1),carrier)
    omega=np.zeros(D,complex);omega[0]=1;omega=Wfull@omega
    U=circuit_to_dense(nsites,gates)
    res=float(np.linalg.norm(U@omega-omega))
    transformed=[]
    for op,ss in ops:
        if 0 in ss or 1 in ss:
            op2=W@embed(op,ss,(0,1))@W.conj().T;transformed.append((op2,(0,1)))
        else:transformed.append((op,ss))
    V=np.stack([embed(O,S,carrier)@omega for O,S in transformed],axis=1)
    def reduced(sites):
        rest=tuple(k for k in carrier if k not in sites)
        tensor=omega.reshape((2,)*nsites)
        mat=np.transpose(tensor,tuple(sites)+rest).reshape(2**len(sites),-1)
        return mat@mat.conj().T
    folded,log=folded_rank6_moments(nsites,gates,transformed,reduced,nmax,1.0,max_cone_sites=6)
    Wstate=V.copy();ref=[]
    for _ in range(nmax+1):
        ref.append(V.conj().T@Wstate);Wstate=U@Wstate
    error=float(np.max(np.abs(folded-np.array(ref))))
    r01=reduced((0,))
    purity=float(np.real(np.trace(r01@r01)))
    if error>1e-11 or res>1e-12 or purity>=1-1e-9:
        raise AssertionError('entangled Kato marginal identity regression')
    return {'classification':'ENTANGLED_EVEN_PARITY_GATE_CONTROL_ONLY',
            'reference_eigenstate_residual':res,'max_abs_moment_error':error,
            'single_site_reduced_purity':purity,'depth':nmax,
            'skipped_disjoint_gates':int(sum(row['skipped_gates'] for row in log))}


def nonstationary_transition_control_test(nsites:int=6,nmax:int=3)->dict:
    """Full transition density, no Floquet-eigenstate hypothesis."""
    ops=[(np.array([[0,0],[1,0]],complex)* (np.sqrt(2) if j==0 else 1.0),(j,)) for j in range(6)]
    gates=[LocalGate((a,a+1),exchange(.14+.037*a,.12*a)) for a in (0,2,4,1,3)]
    full=tuple(range(nsites));D=2**nsites
    O=np.zeros(D,complex);O[0]=1/np.sqrt(2);O[2**(nsites-1)]=1/np.sqrt(2)
    U=circuit_to_dense(nsites,gates)
    residual=float(np.linalg.norm(U@O-O))
    if residual<.01:raise AssertionError('test reference unexpectedly stationary')
    V=np.stack([embed(op,S,full)@O for op,S in ops],axis=1)
    vac=O.reshape((2,)*nsites)
    evol=[O]
    for _ in range(nmax):evol.append(U@evol[-1])
    def transition(n,sites):
        rem=tuple(i for i in full if i not in sites)
        perm=tuple(sites)+rem
        u=np.transpose(evol[n].reshape((2,)*nsites),perm).reshape(2**len(sites),-1)
        v=np.transpose(vac,perm).reshape(2**len(sites),-1)
        return u@v.conj().T
    folded,_=folded_rank6_moments(nsites,gates,ops,None,nmax,None,
                                    max_cone_sites=6,transition_oracle=transition)
    direct=[];W=V.copy()
    for _ in range(nmax+1):direct.append(V.conj().T@W);W=U@W
    err=float(np.max(np.abs(folded-np.asarray(direct))))
    if err>1e-11 or np.linalg.norm(V.conj().T@V-np.eye(6))>1e-11:
        raise AssertionError('nonstationary folded transition identity regression')
    return {'classification':'NONSTATIONARY_COHERENT_TRANSITION_CONTROL_ONLY',
            'reference_floquet_residual':residual,'max_abs_moment_error':err,'depth':nmax}


def independent_control_test(nsites:int=6,nmax:int=4)->dict:
    # Six local source operators O_j=|1><0|_j, with V†V=I_6.
    ops=[(np.array([[0,0],[1,0]],complex),(j,)) for j in range(6)]
    gates=[LocalGate((a,a+1),exchange(.17+.061*a,.09*a)) for a in (0,2,4,1,3)]
    for a in range(5):gates[a].validate(nsites)
    allsites=tuple(range(nsites));dim=2**nsites
    vacuum=np.zeros(dim,complex);vacuum[0]=1
    U=circuit_to_dense(nsites,gates);ground_res=float(np.linalg.norm(U@vacuum-vacuum))
    V=np.stack([embed(op,s,allsites)@vacuum for op,s in ops],axis=1)
    source_res=float(np.linalg.norm(V.conj().T@V-np.eye(6)))
    def rho(sites):
        out=np.zeros((2**len(sites),2**len(sites)),complex);out[0,0]=1
        return out
    F,log=folded_rank6_moments(nsites,gates,ops,rho,nmax,1.0,max_cone_sites=6)
    W=V.copy();direct=[]
    for n in range(nmax+1):
        direct.append(V.conj().T@W);W=U@W
    direct=np.array(direct)
    residual=float(np.max(np.abs(F-direct)))
    gram=float(np.linalg.norm(F[0]-np.eye(6)))
    T=toeplitz(F)
    toeplitz_min=float(np.linalg.eigvalsh(T).min())
    result=resolvent_from_positive_moments(F,2.0+0.5j)
    invnorm=result['resolvent_inverse_norm'];inv_bound=result['inverse_norm_bound']
    if residual>5e-12 or gram>5e-12 or source_res>5e-12 or ground_res>5e-12:
        raise AssertionError('folded/source orbit identity fails')
    if toeplitz_min< -5e-12 or invnorm>inv_bound+5e-12:
        raise AssertionError('positive measure/sine-taper checks failed')
    if not any(row['skipped_gates']>0 for row in log[1:]):
        raise AssertionError('causal support has not removed a gate')
    # Probe non-uniqueness of coherence using a genuine C10 edge probability.
    source_file=Path(__file__).resolve().parents[1]/'data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    from collections import defaultdict
    with np.load(source_file,allow_pickle=False) as z:
        rows=defaultdict(list)
        for a,b,p in zip(z['c10_edge_src'],z['c10_edge_dst'],z['c10_edge_probability']):
            if 1e-9<p<1-1e-9:rows[int(a)].append(float(p))
        p=next((p for ps in rows.values() for p in ps if .1<p<.9),None)
    if p is None:raise AssertionError('no nontrivial C10 edge found')
    # Same population transition P, distinct unitary/kraus coherences.
    c=np.sqrt(p);s=np.sqrt(1-p)
    Uq=np.array([[c,s],[s,-c]],complex)
    P=np.array([[p,1-p],[1-p,p]])
    plus=np.ones(2,complex)/np.sqrt(2);rr=np.outer(plus,plus.conj())
    decoh=np.diag(P@np.real(np.diag(rr)))
    coherent=Uq@rr@Uq.conj().T
    distance=float(.5*np.linalg.svd(coherent-decoh,compute_uv=False).sum())
    if distance<.1:raise AssertionError('C10 population data unexpectedly fixes quantum coherence')
    return {'classification':'ALGORITHMIC_TEST_ONLY_NOT_PHYSICAL_G2',
      'source_action_hash':'c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7',
      'test_circuit_sites':nsites,'depth':nmax,'source_order':['L1','L2','L3','R1','R2','R3'],
      'folded_vs_full_unitary_max_abs_difference':residual,
      'source_gram_identity_fro_residual':source_res,
      'vacuum_eigenstate_residual':ground_res,
      'toeplitz_min_eigenvalue':toeplitz_min,
      'positive_taper_inverse_norm':invnorm,'positive_taper_bound':inv_bound,
      'gate_elimination_statistics':log,
      'entangled_state_cross_check':entangled_state_control_test(),
      'nonstationary_transition_cross_check':nonstationary_transition_control_test(),
      'real_L24_C10_edge_probability_for_nonuniqueness_witness':float(p),
      'two_quantum_channels_same_population_trace_distance':distance,
      'physical_payload_promoted':False,
      'interpretation':'Folded exact-cone contraction matches full-unitary six-column moments; actual C10 population weights do not determine the coherent charged gate.'}


def main():
    obj=independent_control_test()
    out=Path(__file__).resolve().parents[1]/'results/G2_FOLDED_CONE_CONTROL.json'
    out.write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'status':'PASS_FOLDED_CONE_CONTROL','depth':obj['depth'],
                      'max_abs_difference':obj['folded_vs_full_unitary_max_abs_difference'],
                      'gate_skips':sum(x['skipped_gates'] for x in obj['gate_elimination_statistics'])},indent=2))


if __name__=='__main__':main()
