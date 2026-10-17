#!/usr/bin/env python3
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import mpmath as mp

mp.mp.dps = 80
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results'/'microscopic_closure'/'RANK6_TENSOR_STAGE_A_CERTIFICATE.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

L=2; V=L**3; E=3*V; P=3*V; N_MOM=10
PHI=mp.pi/63; X1=4*PHI; X2=32*PHI
EPS_TOTAL=mp.mpf('1e-6')

# Periodic split-step complex.
links=[(mu,x,y,z) for mu in range(3) for x in range(L) for y in range(L) for z in range(L)]
li={t:i for i,t in enumerate(links)}
plaquettes=[]
incidence=[[0]*E for _ in range(P)]
p=0
for mu in range(3):
  for nu in range(mu+1,3):
    for x in range(L):
      for y in range(L):
        for z in range(L):
          s=[x,y,z]; sm=s.copy(); sm[mu]=(sm[mu]+1)%L; sn=s.copy(); sn[nu]=(sn[nu]+1)%L
          po=[(li[(mu,*s)],+1),(li[(nu,*sm)],+1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)]
          plaquettes.append(po)
          for e,sgn in po: incidence[p][e]=sgn
          p+=1
assert len(plaquettes)==24 and all(sum(v!=0 for v in row)==4 for row in incidence)
assert all(sum(incidence[p][e]!=0 for p in range(P))==4 for e in range(E))

# Rank of integer boundary matrix over R is enough for the topology certificate here.
import numpy as np
B=np.array(incidence,dtype=float).T
rank_d2=int(np.linalg.matrix_rank(B,tol=1e-10))
b1=E-V+1-rank_d2+rank_d2  # graph b1 = E-V+1 = 17 for the 1-skeleton
# Cellular H1 of the 3-torus: ker d1 / im d2 = (E-V+1)-rank d2 = 3.
cell_b1=(E-V+1)-rank_d2

# Character tails. We bound the sup norm of one plaquette error, then telescope
# the product of 24*N_MOM norm-one exact plaquette multipliers.
def u1_tail(N:int):
    return 2*mp.nsum(lambda k: abs(mp.besselj(k,X1)), [N+1, mp.inf])

def su2_coeff_abs(j2:int):
    d=j2+1
    return abs(2*d/X2*mp.besselj(d,X2))

def su2_uniform_tail(J2:int):
    # |chi_j(U)| <= 2j+1 = d.
    return mp.nsum(lambda k: abs(2*(k+1)/X2*mp.besselj(k+1,X2))*(k+1), [J2+1, mp.inf])

def accumulated(delta):
    return mp.e**((P*N_MOM)*delta)-1

# Choose the first cutoffs that put each analytic sector below 5e-8,
# leaving > 9e-7 for tensor compression and floating-point certification.
U1_N=None
for N in range(0,30):
    if accumulated(u1_tail(N)) < mp.mpf('5e-8'):
        U1_N=N; break
SU2_J2=None
for J2 in range(0,40):
    if accumulated(su2_uniform_tail(J2)) < mp.mpf('5e-8'):
        SU2_J2=J2; break
assert U1_N is not None and SU2_J2 is not None

eu1=accumulated(u1_tail(U1_N)); esu2=accumulated(su2_uniform_tail(SU2_J2))
remaining=EPS_TOTAL-eu1-esu2
assert remaining > mp.mpf('9e-7')

# SU(2) fusion multiplicity for an ordered list of twice-spins.
def fusion_distribution(spins2):
    dist={0:1}
    for s in spins2:
        nxt=defaultdict(int)
        for a,mult in dist.items():
            for b in range(abs(a-s),a+s+1,2):
                nxt[b]+=mult
        dist=dict(nxt)
    return dist

def inv_mult(spins2):
    return fusion_distribution(spins2).get(0,0)

# Link Haar projector multiplicity: four plaquette representations, and optionally
# a fundamental matter line. Enumerate the certified plaquette cutoff.
max_pure=(0,None); max_left=(0,None)
for a in range(SU2_J2+1):
  for b in range(SU2_J2+1):
    for c in range(SU2_J2+1):
      for d in range(SU2_J2+1):
        m=inv_mult([a,b,c,d])
        if m>max_pure[0]: max_pure=(m,[a,b,c,d])
        ml=inv_mult([a,b,c,d,1])
        if ml>max_left[0]: max_left=(ml,[a,b,c,d,1])

# Peter-Weyl storage dimensions if one ignored symmetry (reported only as a warning).
pw_dim=sum((j2+1)**2 for j2 in range(SU2_J2+1))

# Worst-case *exact support* after n macro-steps of the truncated local gates.
# Each spatial link belongs to four plaquettes. Matter multiplication can shift U(1)
# electric flux by at most |q_R|=6 and SU(2) spin by 1/2 per macro-step.
qmax=6
u1_Emax=N_MOM*(4*U1_N+qmax)
su2_link_Jmax=mp.mpf(N_MOM)*(4*mp.mpf(SU2_J2)/2 + mp.mpf('0.5'))
su2_link_J2max=int(2*su2_link_Jmax)
su2_link_block_count=su2_link_J2max+1
su2_link_dense_pw_dim=sum((k+1)**2 for k in range(su2_link_J2max+1))

report={
 'status':'PASS_STAGE_A_EXACT_LOCAL_TENSOR_AND_CERTIFIED_CHARACTER_CUTOFFS',
 'external_particle_data_used':False,
 'geometry':{
   'L':L,'vertices':V,'links':E,'plaquettes':P,'link_plaquette_valence':4,
   'boundary_d2_rank':rank_d2,'cellular_b1':cell_b1,
   'statement':'Periodic split-step 2x2x2 cubic complex used by C8; the cellular first Betti number is 3.'
 },
 'plaquette_character_expansions':{
   'U1':{
     'x':'4*pi/63','formula':'exp(i x cos(theta)) = sum_k i^k J_k(x) exp(i k theta)',
     'shift_cutoff_abs_k':U1_N,'one_plaquette_uniform_tail':str(u1_tail(U1_N)),
     '240_gate_telescoped_bound':str(eu1),'local_character_blocks':2*U1_N+1
   },
   'SU2':{
     'x':'32*pi/63','formula':'c_j(x)=i^(2j) 2(2j+1) J_(2j+1)(x)/x',
     'plaquette_j_max':str(mp.mpf(SU2_J2)/2),'twice_j_max':SU2_J2,
     'one_plaquette_uniform_tail_using_|chi_j|<=2j+1':str(su2_uniform_tail(SU2_J2)),
     '240_gate_telescoped_bound':str(esu2),'irrep_blocks':SU2_J2+1,
     'naive_Peter_Weyl_dimension_sum_dj2':pw_dim
   },
   'combined_analytic_gate_error_bound':str(eu1+esu2),
   'remaining_absolute_error_budget_for_TRG_and_roundoff':str(remaining)
 },
 'local_link_tensor':{
   'U1_constraint':'delta_{sum_{p contains e} epsilon_pe k_p + q_e J_e, E_out-E_in}; in the static Haar limit E_out=E_in this is Gauss conservation.',
   'SU2_Haar_projector':'H_e = integral dU tensor_a D^{j_a}(U^{epsilon_a}) tensor D^{r_e}(U) = sum_iota |iota><iota| after dual-orientation maps.',
   'max_pure_gauge_intertwiner_multiplicity':max_pure[0],
   'pure_gauge_argmax_twice_spins':max_pure[1],
   'max_left_matter_intertwiner_multiplicity':max_left[0],
   'left_matter_argmax_twice_spins':max_left[1],
   'right_matter_representation':'SU(2) singlet',
   'left_matter_representation':'SU(2) fundamental, twice-spin 1'
 },
 'CAR':{
   'rule':'Gamma_CAR(U)=direct_sum_n wedge^n U',
   'vacuum_two_point_reduction':'Exact restriction to wedge^1 H; no Fock truncation is needed for C_n^(6).',
   'fermion_parity_bond':'A Z2 parity label is retained by the compiler; in the one-particle vacuum correlator exactly one matter worldline connects source and sink.'
 },
 'moment_network':{
   'moments':'C_n^(6)=V_6^dagger U_micro^n V_6, n=0,...,10',
   'time_slices':N_MOM,
   'electric_gates':'Diagonal in U(1) electric flux and SU(2) Peter-Weyl spin; they are kept exactly on the reached support.',
   'plaquette_gates':'Character-expanded shift/fusion tensors using the certified cutoffs above.',
   'matter_gate':'C8 split-step Sx Sy Sz with q_L=-3, q_R=-6 and SU(2)_L fundamental/singlet transport.',
   'rank_six_boundary':'Three family channels times L/R chirality; spin is internally contracted, not fitted.'
 },
 'worst_case_reached_support_for_the_truncated_10_step_circuit':{
   'U1_abs_electric_flux_max':u1_Emax,'U1_time_bond_blocks':2*u1_Emax+1,
   'SU2_link_j_max':str(su2_link_Jmax),'SU2_link_twice_j_max':su2_link_J2max,
   'SU2_link_irrep_blocks':su2_link_block_count,
   'SU2_naive_dense_Peter_Weyl_dimension':su2_link_dense_pw_dim,
   'warning':'These are rigorous worst-case supports, not recommended dense dimensions. The contraction must remain symmetry-block sparse and truncate only Schmidt/intertwiner sectors with an a posteriori norm bound.'
 },
 'compression_rule':{
   'algorithm':'symmetry-preserving higher-order SVD / boundary-MPS elimination on fused U(1) charge, SU(2) irrep/intertwiner, CAR parity blocks',
   'required_a_posteriori_bound':f'sum of discarded spectral norms over all compression events < {mp.nstr(remaining,18)}',
   'promotion_test':'Repeat with tighter character cutoffs and half the compression tolerance; every entry of C_n^(6), the singular spectrum of T_+, and K_UV must change by <1e-6.'
 }
}
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
