#!/usr/bin/env python3
"""Recompute reduced ratios, L24 fixture edges, volume bounds and finite responses.

Interval operations use exact rational arithmetic and outward rounding to a
decimal grid. Reported intervals cover the stated inputs and arithmetic;
they do not bound the physical difference between a reduced and full model.
"""
from fractions import Fraction as F
from math import isqrt,comb,factorial
from pathlib import Path
import csv,hashlib,json,platform
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/quantitative_comparison'
SCALE=10**60

def floorq(x):return F(x.numerator*SCALE//x.denominator,SCALE)
def ceilq(x):return -floorq(-x)
class I:
 def __init__(self,a,b=None):self.a=floorq(F(a));self.b=ceilq(F(a if b is None else b));assert self.a<=self.b
 def __add__(self,o):
  o=as_i(o);return I(self.a+o.a,self.b+o.b)
 __radd__=__add__
 def __neg__(self):return I(-self.b,-self.a)
 def __sub__(self,o):return self+-as_i(o)
 def __rsub__(self,o):return as_i(o)+-self
 def __mul__(self,o):
  o=as_i(o);v=[x*y for x in [self.a,self.b] for y in [o.a,o.b]];return I(min(v),max(v))
 __rmul__=__mul__
 def __truediv__(self,o):
  o=as_i(o);assert not o.a<=0<=o.b;return self*I(1/o.b,1/o.a)
 def __rtruediv__(self,o):return as_i(o)/self
 def sqrt(self):
  assert self.a>=0
  def lower(x):return F(isqrt(x.numerator*SCALE*SCALE//x.denominator),SCALE)
  return I(lower(self.a),lower(self.b)+F(1,SCALE))
 def power(self,n):
  out=I(1)
  for _ in range(n):out=out*self
  return out
 def pair(self):return [decimal(self.a),decimal(self.b)]
 def mid(self):return float((self.a+self.b)/2)
def as_i(x):return x if isinstance(x,I) else I(x)
def decimal(x):
 assert (x*SCALE).denominator==1
 n=int(x*SCALE);sign='-' if n<0 else '';n=abs(n)
 return sign+str(n//SCALE)+'.'+str(n%SCALE).zfill(60)
def atan_interval(q,n):
 x=F(1,q);s=sum((-1)**k*x**(2*k+1)/(2*k+1) for k in range(n));r=(-1)**n*x**(2*n+1)/(2*n+1)
 return I(min(s,s+r),max(s,s+r))
PI=16*atan_interval(5,60)-4*atan_interval(239,20)
def cos_small(x):
 term=I(1);s=term
 for k in range(1,26):term=term*x*x/((2*k-1)*(2*k));s=s+(-term if k%2 else term)
 tail=term*x*x/(51*52)
 return I(s.a,s.b+tail.b)
def minus_log_one_minus(x):
 assert 0<=x.a<x.b<1
 term=I(1);s=I(0)
 for n in range(1,41):term=term*x;s=s+term/n
 tail=term*x/(41*(1-x))
 return I(s.a,s.b+tail.b)
def exp_positive(x):
 assert 0<=x.a<=x.b<20
 term=I(1);s=term
 for n in range(1,201):term=term*x/n;s=s+term
 tail=(term*x/201)/(1-x/202)
 return I(s.a,s.b+tail.b)
def asin_positive(x):
 assert 0<=x.a<=x.b<F(1,2)
 s=I(0)
 for n in range(81):s=s+F(comb(2*n,n),4**n*(2*n+1))*x.power(2*n+1)
 tail=F(comb(162,81),4**81*163)*x.power(163)/(1-x*x)
 return I(s.a,s.b+tail.b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((ROOT/name).read_text())
def cf(x):return (F(float(np.real(x))),F(float(np.imag(x))))
def cmul(x,y):return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 # AGM bounds the elliptic integral exactly: R_BCC=1/(8 AGM(1,1/sqrt(2))^2).
 a,b=I(1),(I(1)/2).sqrt()
 for _ in range(12):a,b=(a+b)/2,(a*b).sqrt()
 agm=I(b.a,a.b);rbcc=1/(8*agm*agm)
 rotor=read('reproducibility/results/CERTIFIED_SU2_ROTOR.json')
 ratio_rows=[]
 for row in rotor['certificates']:
  w=I(*row['wilson_expectation_interval']);t=rbcc*w.sqrt()*cos_small(PI/63)
  alpha=t+minus_log_one_minus(t.power(3))/96
  ratios={name:exp_positive(k*alpha) for name,k in [('mu_e',34),('tau_mu',18),('tau_e',52)]}
  ratio_rows.append({'rotor_cutoff_N':row['N'],'alpha_interval':alpha.pair(),
   'reduced_ratio_intervals':{k:v.pair() for k,v in ratios.items()},
   'ratio_midpoints':{k:v.mid() for k,v in ratios.items()},
   'scope':'Certified local rotor input propagated through the reduced formula; N is not spatial L.'})
 final=ratio_rows[-1]
 # Exact integer moments: this named L24 file is a synthetic two-atom fixture.
 with np.load(ROOT/'reproducibility/data/physical_source/CN6_MOMENTS_L24.npz',allow_pickle=False) as d:
  C=d['C'];u=np.diag([1,1,1,-1,-1,-1]).astype(np.complex128)
  for n,c in enumerate(C):assert np.array_equal(c,np.linalg.matrix_power(u,n))
  assert np.array_equal(d['moments'],C) and np.array_equal(d['direct_moments'],C)
 moment_cert=read('reproducibility/data/physical_source/CN6_ERROR_CERTIFICATE_L24.json')
 assert 'Synthetic' in moment_cert['method']
 l24={'classification':'SYNTHETIC_TWO_ATOM_FIXTURE','moment_count':len(C),
  'eigenvalues_exact':[1,-1],'source_multiplicities':[3,3],
  'quasienergy_lift':'[0,pi] with z=exp(-i epsilon)',
  'quasienergy_intervals':[[0,0],PI.pair()],
  'entrywise_moment_error_exact':0,'C0_equals_identity':True,
  'physical_C10_edge':None,'physical_mass_ratios':None,
  'ratio_to_zero_edge':'undefined','full_moving_C10':False}
 # Read the declared volume schedule. It is a truncation schedule, not masses.
 schedule=read('reproducibility/data/physical_source/VOLUME_SCHEDULE.json')
 volume=[dict(L=int(k),**v) for k,v in schedule['volumes'].items()]
 for row in volume:assert row['nmax']==row['L']//2-3 and row['representation_tail_bound']<1e-6
 bounds=[r['representation_tail_bound'] for r in volume]
 # Finite receiver coordinates, with an explicit serialization-rounding box.
 finite=read('reproducibility/data/FINITE_FLAVOR_BENCHMARKS.json')
 mixing={}
 for sector,key in [('ud','B_ud'),('enu','B_enu')]:
  B=finite[key];error=F(1,2*10**10)
  box=lambda i,j:I(F(str(B[i][j]))-error,F(str(B[i][j]))+error)
  s13=box(0,2).sqrt();s12=(box(0,1)/(1-box(0,2))).sqrt();s23=(box(1,2)/(1-box(0,2))).sqrt()
  angles={k:180*asin_positive(x)/PI for k,x in [('theta12',s12),('theta13',s13),('theta23',s23)]}
  mixing[sector]={'angles_degrees':{k:v.mid() for k,v in angles.items()},
   'serialization_rounding_enclosures_deg':{k:v.pair() for k,v in angles.items()},
   'assumption':'Each serialized B_ij approximates the intended modulus squared within 5e-11. This excludes physical matching and operator errors.',
   'abs_J_serialized':finite['abs_J_'+sector]}
 # Exact rational evaluation of the static gravity-response fixture.
 with np.load(ROOT/'reproducibility/data/SPECTRAL_GRAVITY_SELFTEST_INPUT.npz',allow_pickle=False) as d:
  k=d['K0'];assert np.array_equal(k,np.diag(np.diag(k)))
  eigen=[cf(x)[0] for x in np.diag(k)];assert all(x>eigen[0] for x in eigen[1:])
  vol=d['K1_vol'];A=d['K1_q'];B=d['K1_mq'];K2=d['K2_qmq']
  assert all(np.array_equal(x,x.conj().T) for x in [vol,A,B,K2])
  n=F(float(d['N_cells']));zr=F(float(d['continuum_density_factor']));qell=float(d['qell'])
  eps=zr*cf(vol[0,0])[0]/(4*n);mixed=cf(K2[0,0])[0]
  for j in range(1,len(eigen)):
   mixed+=(cmul(cf(A[0,j]),cf(B[j,0]))[0]+cmul(cf(B[0,j]),cf(A[j,0]))[0])/(eigen[0]-eigen[j])
  chi=zr*mixed/n
  inv=np.diag([0]+[1/float(eigen[0]-e) for e in eigen[1:]])
  numeric=zr*F(float(np.real((K2+A@inv@B+B@inv@A)[0,0])))/n
 gravity={'classification':'FINITE_9_BY_9_SYNTHETIC_SOURCE_RESPONSE',
  'epsilon_source_exact_binary_input_interval':I(eps).pair(),
  'chi_TT_exact_binary_input_interval':I(chi).pair(),
  'chi_independent_float_difference':float(abs(numeric-chi)),
  'gap_exact_binary_input_interval':I(min(x-eigen[0] for x in eigen[1:])).pair(),
  'N_cells':int(n),'qell':qell,
  'physical_graviton_G_Lambda_determined':False,
  'scope':'Rational evaluation of supplied binary64 matrices. No microscopic QICT identification or source-model error enclosure.'}
 # Exact counterexamples underlying the corrected continuum statements.
 moments=[sum(comb(n,j)*(-1)**(n-j)*factorial(2*j)//(2**j*factorial(j)) for j in range(n+1)) for n in range(9)]
 cumulants=[0]
 for n in range(1,9):cumulants.append(moments[n]-sum(comb(n-1,j-1)*cumulants[j]*moments[n-j] for j in range(1,n)))
 assert cumulants[1]==0 and all(cumulants[n]==2**(n-1)*factorial(n-1) for n in range(2,9))
 # All powers of two roots of unity are determined by four residue classes.
 for m in [2,8,64,1024]:
  for r,i in [(1,0),(0,1),(-1,0),(0,-1)]:
   a=F(1,m);assert (a*(1-r))**2+(a*i)**2<=4*a*a
 projector0=np.diag([1,0]);projector1=np.diag([0,1])
 assert np.array_equal((projector0-projector1)@(projector0-projector1),np.eye(2,dtype=int))
 controls={'free_composite_cumulants_1_to_8':cumulants[1:],
  'vanishing_atom':'nu_m=(1/m)delta_1+(1-1/m)delta_i: moments converge, lower lifted support edge changes from 0 to pi/2.',
  'projector_counterexample':'Alternating diag(0,1) and diag(1,0): uniform gap 1, projector distance 1.'}
 exp_mu=I('206.7682827');exp_mu_err=F('0.0000046')
 me=I('0.51099895069');mmu=I('105.6583755');mtau=I('1776.93')
 exp_ratios={'mu_e':exp_mu,'tau_mu':mtau/mmu,'tau_e':mtau/me}
 exp_sigmas={'mu_e':float(exp_mu_err),'tau_mu':float(F('0.09')/F('105.6583755')),'tau_e':float(F('0.09')/F('0.51099895069'))}
 ec=finite['receiver_singular_values']['e'];fin_ratios={'mu_e':ec[1]/ec[0],'tau_mu':ec[2]/ec[1],'tau_e':ec[2]/ec[0]}
 compare=[]
 for key in ['mu_e','tau_mu','tau_e']:
  reduced=final['ratio_midpoints'][key];reference=exp_ratios[key].mid()
  compare.append({'observable':key,'reduced':reduced,'finite_receiver':fin_ratios[key],
   'physical_L24':None,'experimental_reference':reference,'experimental_sigma_approx':exp_sigmas[key],
   'relative_difference_ppm':1e6*(reduced/reference-1),
   'reduced_model_physical_uncertainty':None})
 inputs=['reproducibility/results/CERTIFIED_SU2_ROTOR.json','reproducibility/data/physical_source/CN6_MOMENTS_L24.npz',
 'reproducibility/data/physical_source/CN6_ERROR_CERTIFICATE_L24.json','reproducibility/data/physical_source/VOLUME_SCHEDULE.json',
 'reproducibility/data/FINITE_FLAVOR_BENCHMARKS.json','reproducibility/data/SPECTRAL_GRAVITY_SELFTEST_INPUT.npz',
 'reproducibility/data/physical_source/MULTIVOLUME_SOFTWARE_CHAIN_STATUS.json']
 result={'status':'PASS_QUANTITATIVE_COMPARISON_AND_EXACT_CONTROLS',
 'program_sha256':sha(Path(__file__)),'input_sha256':{p:sha(ROOT/p) for p in inputs},
 'publication_sources_sha256':{p:sha(ROOT/p) for p in [
  'paper/sections/main/m070.tex','paper/sections/main/m115.tex','paper/sections/main/m119.tex',
  'paper/sections/main/m120.tex','paper/sections/main/m175.tex','paper/sections/supplement/s45.tex',
  'paper/sections/supplement/s73.tex','paper/sections/supplement/s75.tex','paper/sections/supplement/s76.tex',
  'paper/sections/supplement/s77.tex','paper/sections/supplement/s148.tex']},
 'environment':{'Python':platform.python_version(),'NumPy':np.__version__},
 'arithmetic':'Exact rational operations with outward rounding on a 1e-60 grid; analytic series remainders.',
 'bcc_return_interval':rbcc.pair(),'rotor_cutoff_stability':ratio_rows,
 'L24_extraction':l24,'spatial_volume_schedule':volume,
 'volume_interpretation':{'all_declared_tail_bounds_below_1e_minus_6':True,'tail_bounds_monotone':all(b<=a for a,b in zip(bounds,bounds[1:])),
 'physical_mass_volume_trend_determined':False,'reason':'The archive supplies no accepted moving charged moments on this volume sequence.'},
 'lepton_comparison':compare,'finite_mixing':mixing,'static_tensor_test':gravity,'logical_controls':controls,
 'experimental_sources':{'lepton_masses':'CODATA 2022, https://physics.nist.gov/cuu/Constants/Table/allascii.txt; PDG 2025 tau listing, https://pdg.lbl.gov/2025/listings/rpp2025-list-tau.pdf',
 'mixing':'NuFIT 6.0 (2024), normal ordering, IC19 without SK atmospheric data, https://www.nu-fit.org/sites/default/files/v60.tbl-parameters.pdf'},
 'statistical_scope':'No chi-square: a full-model prediction covariance and full interacting observables have not been supplied.'}
 (OUT/'QUANTITATIVE_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
 with (OUT/'LEPTON_COMPARISON.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=compare[0]);writer.writeheader();writer.writerows(compare)
 with (OUT/'VOLUME_BOUNDS.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=volume[0]);writer.writeheader();writer.writerows(volume)
 print(json.dumps({'status':result['status'],'ratios':compare,'mixing':{k:v['angles_degrees'] for k,v in mixing.items()},
  'gravity':{'epsilon':float(eps),'chi':float(chi),'independent_float_difference':gravity['chi_independent_float_difference']},
  'physical_L24_spectrum_computed':False},indent=2))

if __name__=='__main__':main()
