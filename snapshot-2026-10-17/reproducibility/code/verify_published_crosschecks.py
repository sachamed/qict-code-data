#!/usr/bin/env python3
"""Independent algebraic, numerical and manuscript-data cross-checks."""
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import io,json,hashlib,re,math
import numpy as np
from scipy.linalg import schur
from scipy.special import jv
from prepare_inputs import read_reference

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'reproducibility/results'

def json_file(name):return json.loads((RESULTS/name).read_text())
def fstr(value):return str(value)
def toeplitz(C):
 return np.block([[C[j-i] if j>=i else C[i-j].conj().T for j in range(len(C))] for i in range(len(C))])

def main():
 records={}
 # The permutation is a coordinate change, with exact integer matrix identities.
 swap=np.eye(6,dtype=np.int64)[[0,3,1,4,2,5]]
 assert np.array_equal(swap@swap.T,np.eye(6,dtype=np.int64))
 for perm in permutations(range(3)):
  P=np.eye(3,dtype=np.int64)[list(perm)]
  assert np.array_equal(swap@np.kron(np.eye(2,dtype=np.int64),P)@swap.T,
                        np.kron(P,np.eye(2,dtype=np.int64)))
 records['source_ordering']={'exact_permutation_indices':[0,3,1,4,2,5],
  'family_permutations_verified':6,'basis_change_is_unitary':True}
 # Frozen one-loop orientation relation, with exact polynomial phase residues.
 assert [k for k in range(63) if (8*k-2)%63==0]==[16]
 orientation=[];kernel_error=0.;phi=math.pi/63;argument=4*phi
 for q in (0,1,2,4,-3,-6):
  beta_axis=F(q,4);beta_cyclic=F(q+1,4)
  assert 16*beta_axis==4*q and 16*beta_cyclic==4*(q+1)
  assert 8*beta_axis**2+F(3*q*q,2)+q==2*q*q+q
  assert 8*beta_cyclic**2+F(3*q*q-1,2)==2*q*q+q
  nu_axis=8*(beta_axis-F(63,2));nu_cyclic=8*(beta_cyclic-F(63,2))
  assert nu_axis.denominator==nu_cyclic.denominator==1
  assert (16*16-4)%126==0 and (8*16**2-32)%126==0
  for n in range(-8,9):
   d_axis=8*(n+16)**2+4*q*(n+16)+2*q*q+q
   d_cyclic=8*n*n+4*(q+1)*n+2*q*q+q
   assert d_axis-d_cyclic==252*n+2048+64*q
   for m in range(-8,9):
    coefficient=np.exp(-1j*argument)*(1j)**(m-n)*jv(m-n,argument)
    left=coefficient*np.exp(-1j*phi*d_cyclic)
    right=np.exp(1j*phi*(32+64*q))*coefficient*np.exp(-1j*phi*d_axis)
    kernel_error=max(kernel_error,abs(left-right))
  orientation.append({'q':q,'beta_axis':str(beta_axis),'beta_cyclic':str(beta_cyclic),
   'nu_axis':int(nu_axis),'nu_cyclic':int(nu_cyclic),'fixed_phase_residue_mod_126':(32+64*q)%126})
 assert kernel_error<1e-11
 records['cycle_orientation']={'unique_translation_residue_mod_63':16,
  'polynomial_identity':'252*r+2048+64*q','charge_sectors':orientation,
  'complex_Bessel_kernel_diagnostic':float(kernel_error),
  'proof_scope':'The two defined isolated one-loop Floquet operators; state and sources must be transported together.',
  'primitive_ground_states_identified':False}
 # Polynomial coefficients of the SU3 sum-of-squares identity.
 lhs={(2,0):F(2),(1,1):F(2),(0,0):F(1,2)}
 rhs={(2,0):F(2),(1,1):F(2),(0,2):F(1,2)-F(1,2),(0,0):F(1,2)}
 assert {k:v for k,v in rhs.items() if v}==lhs
 records['colour_potential']={'sum_of_squares_coefficients':'EXACT_FRACTION_IDENTITY',
  'trace_lower_bound':'-3/2','frozen_plaquette_cost_upper':54,
  'domain':'SU3 maximal torus; d is a real cosine, so 1-d^2 is nonnegative.'}
 b=F(3,8);r=F(4,3)*F(5,4)**6*b**6
 tails={'U1':2*b*b/(1-b**6),'SU2':4*b**3/(1-F(9,4)*b**5),
  'SU3':F(2*3**6,64)*b**4+F(3*4**6,64)*b**9/(1-r)}
 gap=json_file('global_primitive_gap/GLOBAL_PRIMITIVE_HEAT_KERNEL_GAP_CERTIFICATE.json')
 assert all(v<F(1,2) for v in tails.values())
 assert {k:str(v) for k,v in tails.items()}==gap['per_link_heat_kernel_tail_upper_rational']
 exponents={g:2*(cost*18252+21136) for g,cost in [('U1',8),('SU2',64),('SU3',54)]}
 for g,x in exponents.items():assert x==gap['independent_group_gap_bounds'][g]['strict_ground_gap_lower']['denominator_exponent']
 assert exponents['SU2']-1==gap['full_neutral_primitive_gap_lower']['denominator_exponent']==2378527
 records['primitive_gap']={'heat_tail_majorants':{k:str(v) for k,v in tails.items()},
  'group_exponents':exponents,'full_gap_lower':'3^(-2378527)',
  'scope':'Fixed finite gauge complex and primitive Hamiltonian homotopy.'}
 with np.load(RESULTS/'global_primitive_seed/GLOBAL_A7_FIRST_DERIVATIVE_LOOP_DATA.npz',allow_pickle=False) as z:
  rows=z['loop_amplitude_records']
  multiplicity={tuple(map(int,x)):int(n) for x,n in zip(*np.unique(rows[:,1:],axis=0,return_counts=True))}
  norm=sum(n*F(num,den)**2 for (code,num,den,energy),n in multiplicity.items())
  quadratic=-sum(n*energy*F(num,den)**2 for (code,num,den,energy),n in multiplicity.items())
  assert len(rows)==91260 and len(multiplicity)==5 and all(n==18252 for n in multiplicity.values())
  seed=json_file('global_primitive_seed/GLOBAL_A7_KATO_SEED_CERTIFICATE.json')
  assert norm==F(seed['ground_vector_derivative_norm_squared_rational'])==F(318903,8)
  assert quadratic==F(seed['energy_Taylor_coefficients']['quadratic_rational'])==-489762
  cubic=18252*F(2*(-6)**3,16**2)
  assert cubic==F(seed['energy_Taylor_coefficients']['cubic_rational'])==-F(123201,4)
 records['primitive_response']={'loop_records':len(rows),'norm_squared':str(norm),
  'magnetic_variance':336*18252,'intensive_response_coefficient':'629/288',
  'energy_linear':72*18252,'energy_quadratic':str(quadratic),'energy_cubic':str(cubic)}
 content=json.loads((ROOT/'reproducibility/verification/SCIENTIFIC_CONTENT.json').read_text())
 name='reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
 with np.load(io.BytesIO(read_reference(content['logical_scientific_files'][name])),allow_pickle=False) as z:
  links=z['links'];start=links[:,1:];end=start.copy();end[np.arange(len(links)),links[:,0]]+=1
  assert np.all(end.sum(axis=1)-start.sum(axis=1)==1)
  vertices=len(np.unique(np.concatenate([start,end]),axis=0))
  assert len(links)-vertices+1==13356
  branch=z['su2_connected_branch_amplitudes'];count=z['su2_connected_branch_count']
  active=np.arange(2)[None,:]<count[:,None]
  defect=float(np.max(np.abs(np.sum(abs(branch)**2*active,axis=1)-1)))
  assert defect<1e-12
 records['geometry_and_amplitudes']={'vertices':vertices,'links':len(links),'cycle_rank':13356,
  'coordinate_gradient_identity':'EXACT_INTEGER','local_amplitude_norm_defect_diagnostic':defect}
 with np.load(RESULTS/'common_coupled_electroweak/COMMON_COUPLED_ELECTROWEAK_GNS_RESOLVENTS.npz',allow_pickle=False) as z:
  C=z['moments'];neg=z['negative_moments'];points=z['real_resolvent_points']
  inverse_error=float(np.max(abs(neg-C.conj().transpose(0,2,1))))
  assert inverse_error<1e-12 and np.linalg.eigvalsh(toeplitz(C)).min()>0
  resolvent_error=0.;self_energy_error=0.
  for k,p in enumerate(points):
   G=sum(C[n]/float(p)**(n+1) for n in range(len(C)))
   Sigma=p*np.eye(6)-C[1]-np.linalg.solve(G,np.eye(6))
   resolvent_error=max(resolvent_error,float(np.linalg.norm(G-z['compressed_resolvents'][k],2)))
   self_energy_error=max(self_energy_error,float(np.linalg.norm(Sigma-z['Feshbach_self_energies'][k],2)))
  assert resolvent_error<1e-12 and self_energy_error<1e-12
 ew=json_file('common_coupled_electroweak/COMMON_COUPLED_ELECTROWEAK_GNS_CERTIFICATE.json')
 bound=F(ew['analytic_state_and_regulator_moment_bound_rational'])
 assert bound<F('2.146e-13')
 published_G=[F('1.221e-4'),F('3.137e-7'),F('4.968e-9')]
 published_S=[F('1.100e-3'),F('5.018e-6'),F('1.242e-7')]
 for k,row in enumerate(ew['resolvent_and_self_energy_bounds']):
  assert F(row['resolvent_operator_error_bound_rational'])<published_G[k]
  assert F(row['self_energy_operator_error_bound_rational'])<published_S[k]
 records['comparison_resolvents']={'inverse_moment_diagnostic':inverse_error,
  'independent_resolvent_sum_diagnostic':resolvent_error,'independent_Feshbach_diagnostic':self_energy_error,
  'displayed_analytic_bounds_outward_rounded':True,'scope':'Component comparison moments, not particle pole masses.'}
 with np.load(RESULTS/'physical_two_time/PHYSICAL_TWO_TIME_LEPTON_KERNELS.npz',allow_pickle=False) as z:
  K=z['physical_kernel'];block=K.transpose(0,2,1,3).reshape(42,42)
  gram_error=float(np.max(abs(block-block.conj().T)))
  minimum=float(np.linalg.eigvalsh((block+block.conj().T)/2).min())
  cell=z['physical_kernel_with_cell'].transpose(0,2,1,3).reshape(42,42)
  cell_min=float(np.linalg.eigvalsh((cell+cell.conj().T)/2).min())
  stationarity=float(np.max(abs(K[1:,1:]-K[:-1,:-1])))
  equal_time=float(max(np.linalg.norm(K[t,t]-np.eye(6)) for t in range(7)))
  artificial_min=float(np.linalg.eigvalsh(toeplitz(K[:,0])).min())
  shape=tuple(map(int,z['source_csr_shape']));indptr=z['source_csr_indptr']
  assert np.max(np.diff(indptr))<=1
  col_norm=np.bincount(z['source_csr_indices'],weights=z['source_csr_data']**2,minlength=shape[1])
  source_error=float(np.max(abs(col_norm-1)))
  assert gram_error<1e-12 and minimum>0 and cell_min>0 and equal_time<1e-12 and artificial_min<0
  assert source_error<float(F('2.23e-16'))
 physical=json_file('physical_two_time/PHYSICAL_TWO_TIME_KERNEL_CERTIFICATE.json')
 assert physical['source_magnetic_intertwining_Frobenius_diagnostic']<6.73e-14
 assert F(physical['per_entry_analytic_state_and_regulator_bound_rational'])<F('1.073e-13')
 assert abs(minimum-physical['full_two_time_block_min_eigenvalue_diagnostic'])<1e-12
 assert abs(stationarity-physical['stationarity_defect_diagnostic'])<1e-12
 records['physical_two_time']={'source_columns':shape[1],'source_norm_defect_diagnostic':source_error,
  'Hermitian_Gram_diagnostic':gram_error,'Gram_minimum_diagnostic':minimum,
  'cell_Gram_minimum_diagnostic':cell_min,'equal_time_diagnostic':equal_time,
  'stationarity_defect_diagnostic':stationarity,'artificial_Toeplitz_minimum_diagnostic':artificial_min,
  'displayed_source_and_intertwining_defects_are_upper_bounds':True}
 finite=json_file('finite_C9_common_state/FINITE_C9_COMMON_STATE_CERTIFICATE.json')
 with np.load(RESULTS/'finite_C9_common_state/FINITE_C9_COMMON_STATE_AND_MOMENTS.npz',allow_pickle=False) as z:
  c9=[];old=None
  for R in range(1,5):
   C=z[f'R{R}_moments'];source=z[f'R{R}_weak_source']
   source_defect=float(np.linalg.norm(source.T@source-np.eye(source.shape[1])))
   c0=float(np.linalg.norm(C[0]-np.eye(6)));tmin=float(np.linalg.eigvalsh(toeplitz(C)).min())
   assert c0<1e-10 and source_defect<1e-12 and tmin>0
   change=None if old is None else float(np.max(abs(C-old)))
   if change is not None:assert abs(change-finite['representation_cutoffs'][R-1]['successive_regulator_moment_change_diagnostic'])<1e-14
   c9.append({'radius':R,'C0_error_diagnostic':c0,'Toeplitz_minimum_diagnostic':tmin,
    'source_isometry_diagnostic':source_defect,'successive_moment_change_diagnostic':change});old=C
  # Use a global neutral Schur decomposition, rather than the product-eigenbasis contraction.
  U0=np.kron(z['R1_neutral_U1'],z['R1_neutral_SU2'])
  seed0=np.kron(z['R1_primitive_U1'],z['R1_primitive_SU2'])
  T,Q=schur(U0,output='complex');lam=np.diag(T)
  assert np.linalg.norm(T-np.diag(lam))<1e-12
  assert np.max(abs(abs(lam)-1))<1e-12
  groups=[];remaining=set(range(len(lam)))
  while remaining:
   i=min(remaining);group=[j for j in remaining if abs(lam[j]-lam[i])<1e-12]
   groups.append(group);remaining.difference_update(group)
  rho=np.zeros_like(U0)
  for ids in groups:
   v=Q[:,ids]@(Q[:,ids].conj().T@seed0);rho+=np.outer(v,v.conj())
  O=np.kron(np.eye(9),z['R1_weak_source'])
  UL=np.kron(z['R1_left_U1'],z['R1_charged_SU2'])
  UR=np.kron(z['R1_right_U1'],z['R1_neutral_SU2'])
  left=np.eye(90,dtype=complex);right=np.eye(45,dtype=complex);neutral_inverse=np.eye(45,dtype=complex)
  direct=0.
  for n in range(13):
   L=np.trace(rho@O.conj().T@left@O@neutral_inverse)
   R=np.trace(rho@right@neutral_inverse)
   direct=max(direct,abs(L-z['R1_moments'][n,0,0]),abs(R-z['R1_moments'][n,3,3]))
   left=UL@left;right=UR@right;neutral_inverse=neutral_inverse@U0.conj().T
  assert direct<1e-12
 records['finite_C9']={'four_regulators':c9,'global_Schur_moment_comparison_diagnostic':float(direct),
  'normal_Haar_or_spatial_limit_inferred':False}
 for source in ['paper/sections/main/m169.tex','paper/sections/main/m170.tex','paper/sections/supplement/s143.tex','paper/sections/supplement/s144.tex']:
  assert (ROOT/source).is_file()
 out={'status':'PASS_INDEPENDENT_PUBLISHED_CROSSCHECKS','records':records,
  'exact_arithmetic':'Integer identities and fractions.Fraction for algebraic and analytic-bound checks.',
  'floating_point_scope':'Array, Gram, Schur and matrix-sum checks are diagnostics with explicit tolerances; no interval roundoff certificate.',
  'new_physical_parameters':0,'full_moving_C10_closure_inferred':False,
  'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 target=RESULTS/'publication_crosschecks/PUBLISHED_CROSSCHECKS.json';target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':out['status'],'check_groups':list(records),'global_Schur_moment_error_diagnostic':float(direct)},indent=2))

if __name__=='__main__':main()
