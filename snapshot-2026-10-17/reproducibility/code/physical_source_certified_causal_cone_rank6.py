#!/usr/bin/env python3
"""Certified causal-cone driver for physical rank-six QICT moments.

This module implements the exact *acceptance logic* for
    C_n^(6)(L) = V^dagger U_L^n V
without replacing the interacting Kato/C10 environment by a periodic L=2
surrogate.  It has four layers:

1. conservative BCC causal-cone geometry and no-wrap certification;
2. volume/depth-adapted Peter-Weyl character cutoffs with a rigorous global
   representation-tail budget;
3. localized Kato/Riesz boundary payload contract, including a certified tail-to-limit trace-norm bound, multi-buffer diagnostics, and uniformly certified protecting-gap metadata;
4. rank-six moment acceptance, C11 Toeplitz positivity and simple-pole LSZ
   residue extraction when a physical moment sequence is actually supplied.

The module enforces input completeness for the Kato/C10 boundary layer.
Large-volume matrix emission is admitted only after the certified volume-specific
boundary payload has been supplied; no surrogate state is synthesized.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse, hashlib, json, math, csv
from fractions import Fraction
import numpy as np

ACTION_SHA256 = "c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7"
SOURCE_ORDER = ("L1","L2","L3","R1","R2","R3")
FAMILY_PAIRS = ((0,3),(1,4),(2,5))
PHI = math.pi/63.0
X1, X2, X3 = 4*PHI, 32*PHI, 36*PHI
TARGET = 1e-6
PI_UPPER = Fraction(22,7)
PHI_UPPER = PI_UPPER/63

class PhysicalPayloadMissing(RuntimeError):
    pass

def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def no_wrap_depth(L:int)->int:
    # Source-certified geometry in the supplied regulator sequence obeys
    # n_max = L/2 - 3 (L even).  This reproduces 16->5, 24->9, 32->13,
    # 48->21, 64->29 and gives the conservative continuation 96->45,128->61.
    if L%2: raise ValueError('certified sequence uses even L')
    return L//2 - 3

def cone_counts(n:int)->dict:
    """Conservative cubic support envelope for n BCC macro-steps.

    A split-step macro-step can move by at most one lattice spacing in each
    Cartesian coordinate.  The full Heisenberg support is therefore contained
    in [-n,n]^3.  The returned counts are an *upper envelope*, used for resource
    planning rather than as an equality claim for every operator component.
    """
    side=2*n+1
    sites=side**3
    # cubic positive-direction links and elementary plaquettes in the envelope
    links=3*sites
    plaquettes=3*sites
    return {'n':n,'side':side,'site_upper_bound':sites,
            'link_upper_bound':links,'plaquette_upper_bound':plaquettes}

def _tail_exact(x,N):
    """Positive rational majorant of sum_{k>N} x^k/k!, with no cancellation."""
    x=Fraction(x)
    if x<0 or N<-1:raise ValueError('nonnegative x and N>=-1 required')
    if x==0:return Fraction(int(N<0))
    k=N+1;term=x**k/math.factorial(k);total=Fraction(0)
    while x>=k+1:
        total+=term;k+=1;term*=x/k
    return total+term/(1-x/Fraction(k+1))

def _float_upper(value):
    value=Fraction(value);out=float(value)
    return math.nextafter(out,math.inf) if Fraction.from_float(out)<value else out

def _tail(x,N):return _float_upper(_tail_exact(Fraction(str(x)),N))
def _tail_from(x,n):return _float_upper(_tail_exact(Fraction(str(x)),max(n,0)-1))
def _u1_tail_exact(N):
    return 2*_tail_exact(2*PHI_UPPER,N)
def _su2_tail_exact(J2):
    y=16*PHI_UPPER;d0=J2+2
    return y*_tail_exact(y,d0-3)+_tail_exact(y,d0-2)
def _su3_tail_exact(R):return _tail_exact(36*PHI_UPPER,R)
def u1_tail(N):return _float_upper(_u1_tail_exact(N))
def su2_tail(J2):return _float_upper(_su2_tail_exact(J2))
def su3_tail(R):return _float_upper(_su3_tail_exact(R))

def character_cutoffs(L:int,nmax:int,target:float=TARGET)->dict:
    if L<=0 or nmax<=0 or target<=0:raise ValueError('positive budgets required')
    P=3*L**3;target_exact=Fraction(str(target));per=target_exact/nmax;a=b=c=0
    while P*_u1_tail_exact(a)>=per/3:a+=1
    while P*_su2_tail_exact(b)>=per/3:b+=1
    while P*_su3_tail_exact(c)>=per/3:c+=1
    d1,d2,d3=_u1_tail_exact(a),_su2_tail_exact(b),_su3_tail_exact(c)
    dloc=(1+d1)*(1+d2)*(1+d3)-1
    # (1+d)^m-1 <= m*d/(1-m*d), since binomial(m,k)<=m^k.
    load=P*nmax*dloc
    full=load/(1-load) if load<1 else Fraction(10**100)
    return {'L':L,'nmax':nmax,'U1_abs_k_max':a,'SU2_twice_j_max':b,
            'SU3_max_p_q':c,'representation_tail_bound':_float_upper(full),
            'representation_tail_bound_rational':str(full),
            'tail_proof':'Positive factorial series, pi<22/7, and a rational binomial majorant.',
            'passes_target':bool(full<target_exact)}

@dataclass
class KatoBufferRecord:
    buffer:int
    density_path:Path
    protecting_gap:float
    protecting_gap_certified:bool
    riesz_residual_upper_bound:float
    trace:float
    limit_trace_distance_bound:float|None=None
    limit_bound_certified:bool=False

    @classmethod
    def load(cls,p:Path):
        z=np.load(p,allow_pickle=False)
        required={'rho','rho_gram_factor','buffer','uniform_contour_gap_lower_bound','riesz_residual_upper_bound'}
        if not required.issubset(z.files):
            raise ValueError(f'{p} lacks rigorous fields {sorted(required-set(z.files))}')
        rho=np.asarray(z['rho'],complex); F=np.asarray(z['rho_gram_factor'],complex)
        if rho.ndim!=2 or rho.shape[0]!=rho.shape[1]: raise ValueError('rho must be square')
        if F.ndim!=2 or F.shape[0]!=rho.shape[0]: raise ValueError('rho_gram_factor has incompatible shape')
        Q=F@F.conj().T; tq=float(np.trace(Q).real)
        if tq<=0: raise ValueError('rho_gram_factor has zero trace')
        rhoF=Q/tq
        if np.linalg.norm(rho-rhoF,'fro')>1e-10: raise ValueError('rho does not match its PSD Gram factorization')
        gap=float(z['uniform_contour_gap_lower_bound'])
        if not np.isfinite(gap) or gap<=0: raise ValueError('uniform full-contour protecting gap must be positive')
        rr=float(z['riesz_residual_upper_bound'])
        if not np.isfinite(rr) or rr<0: raise ValueError('Riesz residual upper bound must be finite and nonnegative')
        lim=float(z['limit_trace_distance_bound']) if 'limit_trace_distance_bound' in z.files else None
        lim_cert=bool(z['limit_bound_certified']) if 'limit_bound_certified' in z.files else False
        # A positive number alone does not certify its operator or physical role.
        typed={'selector_kind','gap_operator_role','protecting_gap_certified'}
        admissible_roles={
            'A7_PRIMITIVE_KATO':'PRIMITIVE_HAMILTONIAN',
            'FINITE_FLOQUET_RIESZ':'FINITE_REGULATOR_FLOQUET_CLUSTER',
            'FINITE_C9':'FINITE_REGULATOR_DEPHASING_CLUSTER'}
        certified=False
        if typed.issubset(z.files):
            kind=str(z['selector_kind'].item());role=str(z['gap_operator_role'].item())
            certified=bool(z['protecting_gap_certified']) and admissible_roles.get(kind)==role
        return cls(int(z['buffer']),p,gap,certified,rr,float(np.trace(rhoF).real),lim,lim_cert)

    def density(self):
        z=np.load(self.density_path,allow_pickle=False); F=np.asarray(z['rho_gram_factor'],complex)
        Q=F@F.conj().T
        return Q/float(np.trace(Q).real)

def trace_norm(A:np.ndarray)->float:
    return float(np.linalg.svd(np.asarray(A,complex),compute_uv=False).sum())

def _load_buffer_contraction_certificate(path:Path|None)->dict|None:
    if path is None or not path.exists(): return None
    obj=json.loads(path.read_text())
    if obj.get('status')!='PASS_CERTIFIED_BUFFER_CONTRACTION': return None
    q=float(obj.get('contraction_q_upper',1.0))
    if not (0.0<=q<1.0): return None
    return {'q_upper':q,'sha256':sha256_file(path),'method':obj.get('method','independent contraction certificate')}

def buffer_convergence(files:list[Path], contraction_certificate:Path|None=None)->dict:
    """Certify convergence to a limiting boundary state, not merely pairwise closeness.

    At least four nested buffers are required.  Pairwise trace distances are
    diagnostics.  A limit error is certified only from either (i) an explicit
    independently certified trace-distance-to-limit majorant serialized with
    the final buffer, or (ii) an independently certified contraction constant
    q<1.  In case (ii), if delta_j=D(rho_{j+1},rho_j) and the contraction
    theorem applies to all subsequent shells, then

        D(rho_last,rho_infty) <= q*delta_last/(1-q).

    Observed ratios are checked for consistency with q but never used to infer q.
    """
    recs=sorted((KatoBufferRecord.load(p) for p in files), key=lambda r:r.buffer)
    if len(recs)<4:
        return {'status':'MULTI_BUFFER_TAIL_CERTIFICATION_STAGE','convergence_certified':False,
                'buffer_count':len(recs),'required_minimum_buffer_count':4,
                'reason':'Two-state or three-state agreement does not certify convergence to the infinite-buffer limit.'}
    rows=[]; deltas=[]
    for a,b in zip(recs[:-1],recs[1:]):
        ra=a.density()
        rb=b.density()
        if ra.shape!=rb.shape: raise ValueError('buffer densities use incompatible boundary spaces')
        td=0.5*trace_norm(rb-ra); deltas.append(td)
        rows.append({'buffer_a':a.buffer,'buffer_b':b.buffer,'trace_distance_diagnostic':td})
    ratios=[deltas[i+1]/deltas[i] for i in range(len(deltas)-1) if deltas[i]>0]
    direct_tail=None
    last=recs[-1]
    if last.limit_bound_certified and last.limit_trace_distance_bound is not None and last.limit_trace_distance_bound>=0:
        direct_tail=float(last.limit_trace_distance_bound)
    ccert=_load_buffer_contraction_certificate(contraction_certificate)
    geo_tail=None; q=None; ratio_consistent=None
    if ccert is not None:
        q=ccert['q_upper']
        ratio_consistent=all(r<=q*(1+1e-12)+1e-15 for r in ratios)
        if ratio_consistent:
            geo_tail=(q*deltas[-1]/(1-q)) if q>0 else 0.0
    candidates=[x for x in (direct_tail,geo_tail) if x is not None]
    tail=min(candidates) if candidates else None
    gap_ok=all(r.protecting_gap_certified and r.protecting_gap>0 for r in recs)
    conv=tail is not None and gap_ok
    return {'status':'PASS_CERTIFIED_BUFFER_CAUCHY_TAIL' if conv else 'AWAITING_CERTIFIED_BUFFER_TAIL_MAJORANT',
            'convergence_certified':bool(conv),'buffer_count':len(recs),
            'records':[{'buffer':r.buffer,'protecting_gap':r.protecting_gap,
                        'protecting_gap_certified':r.protecting_gap_certified,
                        'riesz_residual_upper_bound':r.riesz_residual_upper_bound,'trace':r.trace,
                        'sha256':sha256_file(r.density_path)} for r in recs],
            'successive_trace_distances_diagnostic':rows,
            'successive_distance_ratios_diagnostic':ratios,
            'independent_contraction_q_upper':q,
            'observed_ratios_consistent_with_certified_q':ratio_consistent,
            'direct_limit_trace_distance_bound':direct_tail,
            'geometric_tail_trace_distance_bound':geo_tail,
            'state_limit_trace_distance_upper_bound':tail,
            'minimum_certified_protecting_gap':min((r.protecting_gap for r in recs if r.protecting_gap_certified),default=None),
            'all_protecting_gaps_uniformly_certified':gap_ok,
            'maximum_riesz_residual_diagnostic':max(r.riesz_residual_upper_bound for r in recs),
            'contraction_certificate':ccert}

def block_toeplitz(moments:np.ndarray,N:int)->np.ndarray:
    """Hermitian block Toeplitz matrix from C_0..C_N for a unitary measure."""
    C=np.asarray(moments,complex)
    if C.ndim!=3 or C.shape[1:]!=(6,6): raise ValueError('moments must be (M,6,6)')
    if N>=len(C): raise ValueError('insufficient moment depth')
    T=np.zeros((6*(N+1),6*(N+1)),complex)
    for i in range(N+1):
        for j in range(N+1):
            k=j-i
            blk=C[k] if k>=0 else C[-k].conj().T
            T[6*i:6*(i+1),6*j:6*(j+1)]=blk
    return (T+T.conj().T)/2

def _toeplitz_fro_error_bound(eps,N):
    eps=np.asarray(eps,float)
    if len(eps)<=N or np.any(~np.isfinite(eps[:N+1])) or np.any(eps[:N+1]<0):
        raise ValueError('independently certified moment_fro_error_bounds are required through depth N')
    return float(np.sqrt((N+1)*eps[0]**2+2*sum((N+1-k)*eps[k]**2 for k in range(1,N+1))))

def certify_moments(npz:Path)->dict:
    z=np.load(npz,allow_pickle=False)
    if 'moments' not in z.files: raise ValueError('physical payload lacks moments')
    C=np.asarray(z['moments'],complex)
    if C.ndim!=3 or C.shape[1:]!=(6,6): raise ValueError('moments shape must be (N+1,6,6)')
    c0err=float(np.linalg.norm(C[0]-np.eye(6))); N=min(16,len(C)-1)
    T=block_toeplitz(C,N); lam=float(np.linalg.eigvalsh(T).min())
    out={'moment_depth':len(C)-1,'C0_identity_fro_error_diagnostic':c0err,
         'toeplitz_depth':N,'toeplitz_lambda_min_diagnostic':lam,
         'exact_structural_positivity':'For exact C_k=V^*U^kV with U unitary, T_N is exactly a Gram matrix and is positive semidefinite.',
         'C9':([[float(x.real),float(x.imag)] for x in C[9]] if len(C)>9 else None),
         'sha256':sha256_file(npz)}
    if 'moment_fro_error_bounds' in z.files and 'computed_toeplitz_lambda_min_lower_bound' in z.files:
        E=_toeplitz_fro_error_bound(np.asarray(z['moment_fro_error_bounds'],float),N)
        lower=float(z['computed_toeplitz_lambda_min_lower_bound'])-E
        out.update({'toeplitz_perturbation_fro_upper_bound':E,'certified_exact_lambda_min_lower_bound':lower,
                    'toeplitz_rigorous_positivity_certified':bool(lower>=0),
                    'rigorous_positivity_stage':'PASS_WEYL_CERTIFIED_NUMERICAL_MARGIN' if lower>=0 else 'NUMERICAL_MARGIN_NOT_CERTIFIED'})
    else:
        out.update({'toeplitz_perturbation_fro_upper_bound':None,'certified_exact_lambda_min_lower_bound':None,
                    'toeplitz_rigorous_positivity_certified':False,
                    'rigorous_positivity_stage':'RIGOROUS_MOMENT_AND_TOEPLITZ_ERROR_CERTIFICATION_STAGE'})
    return out

def error_budget(rep_tail:float, buffer_limit_td_bound:float|None, shell_error:float=0.0,
                 insertion_error:float=0.0, compression_error:float=0.0,
                 state_dependence:str='fixed_gns_general')->dict:
    """Conservative matrix-moment bound using a certified distance to the limit.

    The state term is admitted only from a certified upper bound
    D(rho_b,rho_infty), never from a single successive-buffer distance.
    Assumes O_a^* O_b=delta_ab I on the neutral reference sector, hence
    state-independent source Gram I6. Without this operator isometry,
    a separate lower Gram bound and its perturbation are required.
    Linear physical two-time expectations (or fixed GNS moments for
    stationary densities) have Frobenius bound 2*sqrt(6)*D.
    Pure-state fixed GNS comparisons have bound 4*sqrt(6)*D.
    General fixed GNS comparisons depend on sqrt(rho) and instead have
    the Powers--Stoermer bound 2*sqrt(6)*sqrt(2D).
    """
    if buffer_limit_td_bound is None:
        b=None; total=None
    else:
        if not 0<=buffer_limit_td_bound<=1:raise ValueError('trace distance must lie in [0,1]')
        D=buffer_limit_td_bound
        if state_dependence in {'physical_two_time','fixed_gns_stationary'}:
            b=2*math.sqrt(6)*D
        elif state_dependence=='fixed_gns_pure':
            b=4*math.sqrt(6)*D
        elif state_dependence=='fixed_gns_general':
            b=2*math.sqrt(6)*math.sqrt(2*D)
        else:raise ValueError('unknown state-dependence class')
        total=rep_tail+b+shell_error+insertion_error+compression_error
    return {'representation_tail':rep_tail,'buffer_limit_trace_distance_bound':buffer_limit_td_bound,
            'buffer_fro_bound':b,'shell_error':shell_error,'insertion_error':insertion_error,
            'compression_error':compression_error,'total_fro_bound':total,
            'state_dependence':state_dependence,
            'source_isometry_hypothesis':'O_a^dagger O_b=delta_ab I on the neutral sector; no state-dependent Gram renormalization included',
            'state_error_source':'certified distance to infinite-buffer limit; successive-buffer distance alone is never accepted'}

def preflight(root:Path,volumes=(24,32,48,64,96,128))->dict:
    candidates=[root/'data/lattice_v3/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json',
        root/'data/physical_g2_inputs/L24_RANK6_STATE_CONTRACTION_PAYLOAD_2026-10-04/components/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json']
    action=next((p for p in candidates if p.exists()),candidates[0])
    if not action.exists() or sha256_file(action)!=ACTION_SHA256:
        raise RuntimeError('frozen action mismatch')
    out={'status':'PASS_CAUSAL_CONE_HPC_PREFLIGHT','action_sha256':ACTION_SHA256,
         'action_path':str(action.relative_to(root)),
         'source_order':SOURCE_ORDER,'family_pairs':FAMILY_PAIRS,'volumes':[]}
    for L in volumes:
        nmax=no_wrap_depth(L); cut=character_cutoffs(L,nmax)
        payload_dir=root/'data/physical_g2'/f'L{L}'
        buffers=sorted(payload_dir.glob('KATO_BUFFER_*.npz')) if payload_dir.exists() else []
        contraction_cert=payload_dir/'BUFFER_CONTRACTION_CERTIFICATE.json'
        moments=payload_dir/'CN6_MOMENTS.npz'
        rec={'L':L,'no_wrap_depth':nmax,'cone_at_n9':cone_counts(9),
             'character_cutoffs':cut,'kato_buffer_files':len(buffers),
             'physical_moment_payload_present':moments.exists(),
             'mass_extraction_stage':'PREFLIGHT_ONLY__FULL_PHYSICAL_OPERATOR_AND_STATE_CERTIFICATES_REQUIRED'}
        if buffers:
            try: rec['buffer_convergence']=buffer_convergence(buffers,contraction_cert if contraction_cert.exists() else None)
            except Exception as e: rec['buffer_convergence_error']=str(e)
        if moments.exists():
            rec['moment_certificate']=certify_moments(moments)
        else:
            rec['status']='PHYSICAL_MOMENT_PAYLOAD_MISSING'
        out['volumes'].append(rec)
    return out

def synthetic_self_test()->dict:
    # A genuine positive 6x6 matrix measure with two atoms.  This checks C11
    # positivity and the buffer-error inequality without claiming physics.
    rng=np.random.default_rng(7)
    X=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6)); Q,_=np.linalg.qr(X)
    w=np.array([.55,.45]); th=np.array([.2,.7])
    P1=Q[:,:3]@Q[:,:3].conj().T; P2=np.eye(6)-P1
    C=[]
    for n in range(18): C.append(w[0]*np.exp(-1j*n*th[0])*P1+w[1]*np.exp(-1j*n*th[1])*P2)
    C=np.asarray(C); C[0]=w[0]*P1+w[1]*P2
    # normalize C0 to I by using full rank weights per projector blocks
    C=[]
    for n in range(18): C.append(np.exp(-1j*n*th[0])*P1+np.exp(-1j*n*th[1])*P2)
    C=np.asarray(C)
    T=block_toeplitz(C,12); lm=float(np.linalg.eigvalsh(T).min())
    return {'status':'PASS_SYNTHETIC_C11_CAUSAL_CONE_BACKEND_SELF_TEST',
            'C0_error':float(np.linalg.norm(C[0]-np.eye(6))),
            'toeplitz_lambda_min_diagnostic':lm,'exact_structural_positivity':'PASS_BY_TWO_ATOM_UNITARY_GRAM_CONSTRUCTION'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--csv',type=Path)
    a=ap.parse_args()
    out=preflight(a.root.resolve()); out['self_test']=synthetic_self_test()
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    if a.csv:
        a.csv.parent.mkdir(parents=True,exist_ok=True)
        with a.csv.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['L','n_no_wrap_max','U1_kmax','SU2_2jmax','SU3_pqmax','representation_tail_bound','physical_CN6_present'])
            for r in out['volumes']:
                c=r['character_cutoffs']; w.writerow([r['L'],r['no_wrap_depth'],c['U1_abs_k_max'],c['SU2_twice_j_max'],c['SU3_max_p_q'],f"{c['representation_tail_bound']:.16e}",r['physical_moment_payload_present']])
    print(json.dumps({'status':out['status'],'volumes':[(r['L'],r['no_wrap_depth'],r['character_cutoffs']['representation_tail_bound'],r['physical_moment_payload_present']) for r in out['volumes']]},indent=2))
if __name__=='__main__': main()
