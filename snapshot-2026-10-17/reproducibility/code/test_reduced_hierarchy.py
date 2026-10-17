#!/usr/bin/env python3
"""Reproduce conditional Monte Carlo, rotor sensitivity and colour-cost tests.

Sampling probabilities refer to the declared null measures, not a probability
that QICT is correct. Rational residual bounds validate the rotor brackets.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,platform
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.special import betaincinv,gamma,mathieu_b,log_ndtr
from quantitative_comparison import I,PI,cos_small,minus_log_one_minus

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/reduced_hierarchy_tests'
K=float(gamma(.25)**4/(32*np.pi**3)*np.cos(np.pi/63))
NMC=20_000_000
SEED=20261014

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def connect(a):return a-np.log1p(-a**3)/96
def rotor(A,N=48):
 n=np.arange(N+1);e,v=eigh_tridiagonal(n*(n+2)+A,np.full(N,-A/2),lapack_driver='stev')
 u=v[:,0];C=np.diag(np.full(N,.5),1)+np.diag(np.full(N,.5),-1)
 w=float(u@C@u);coeff=v[:,1:].T@C@u
 wp=float(2*np.sum(coeff**2/(e[1:]-e[0])))
 a=K*np.sqrt(w);ap=(1+a*a/(32*(1-a**3)))*a*wp/(2*w)
 return {'N':N,'A':float(A),'energy':float(e[0]),'W':w,'W_derivative':wp,'alpha':float(connect(a)),'alpha_derivative':float(ap),'gap':float(e[1]-e[0])}

def sturm(diag,b,x):
 p0=F(1);p1=diag[0]-x;sg=[1]
 if p1:sg.append(1 if p1>0 else -1)
 for d in diag[1:]:
  p0,p1=p1,(d-x)*p1-b*b*p0
  if p1:sg.append(1 if p1>0 else -1)
 return sum(a!=b for a,b in zip(sg,sg[1:]))

def rational_rotor(A,N=32):
 """Float trial only; infinite-space inequalities use Fraction arithmetic."""
 A=F(A);b=A/2;diag=[F(n*(n+2))+A for n in range(N+1)]
 _,v=eigh_tridiagonal(np.array(diag,dtype=float),np.full(N,-float(b)),lapack_driver='stev')
 u=[F(float(x)) for x in v[:,0]];norm=sum(x*x for x in u)
 hu=[diag[i]*x-b*((u[i-1] if i else 0)+(u[i+1] if i<N else 0)) for i,x in enumerate(u)]
 mu=sum(x*y for x,y in zip(u,hu))/norm
 res=sum((x-mu*y)**2 for x,y in zip(hu,u))+b*b*u[-1]**2
 res/=norm
 lo,hi=F(-1),max(diag)+2*b+1
 for _ in range(90):
  mid=(lo+hi)/2
  if sturm(diag,b,mid)<=1:lo=mid
  else:hi=mid
 assert sturm(diag,b,lo)==1 and sturm(diag,b,hi)==2
 q=F((N+1)*(N+3));lower=min(q/2,lo-2*b*b/q);gap=lower-mu
 assert gap>0
 err=I(res).sqrt()/gap
 w=sum(u[i]*u[i+1] for i in range(N))/norm
 wi=I(w-2*err.b,w+2*err.b)
 a0,b0=I(1),(I(1)/2).sqrt()
 for _ in range(12):a0,b0=(a0+b0)/2,(a0*b0).sqrt()
 ki=cos_small(PI/63)/(8*I(b0.a,a0.b).power(2))
 af=ki*wi.sqrt();alpha=af+minus_log_one_minus(af.power(3))/96
 return {'A':str(A),'N':N,'W_interval':wi.pair(),'alpha_interval':alpha.pair(),
  'gap_lower':str(gap),'projector_error_upper':str(err.b),'residual_squared':str(res),'trial_vector_binary_rationals':[str(x) for x in u]},wi,alpha

def interval_for(t,refs,powers):
 lo=float(np.max(np.log(refs*(1-t))/powers));hi=float(np.min(np.log(refs*(1+t))/powers))
 return lo,hi
def rational_log(x):
 x=F(x);j=0
 while x>=2:x/=2;j+=1
 def unit(y):
  q=(I(y)-1)/(I(y)+1);s=I(0);term=q
  for n in range(100):s=s+term/(2*n+1);term=term*q*q
  tail=2*term/(201*(1-q*q))
  return I(2*s.a,2*s.b+tail.b)
 return unit(x)+j*unit(F(2))
def cp(k,n):
 return [0. if k==0 else float(betaincinv(k,n-k+1,.025)),1. if k==n else float(betaincinv(k+1,n-k,.975))]

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 prior=json.loads((ROOT/'reproducibility/results/quantitative_comparison/QUANTITATIVE_COMPARISON.json').read_text())
 alpha=float(sum(map(F,prior['rotor_cutoff_stability'][-1]['alpha_interval']))/2)
 refs=np.array([206.7682827,1776.93/105.6583755,1776.93/.51099895069]);powers=np.array([34.,18.,52.])
 predicted=np.exp(powers*alpha);T=float(np.max(np.abs(predicted/refs-1)))
 tolerances={'observed_max_relative_discrepancy':T,'one_per_mille':.001}
 intervals={name:interval_for(t,refs,powers) for name,t in tolerances.items()}
 trials=[]
 for j,upper in enumerate([float(connect(K)),1.]):
  rng=np.random.Generator(np.random.PCG64(SEED+j));counts={k:0 for k in tolerances}
  for _ in range(NMC//500000):
   samples=rng.uniform(0,upper,500000)
   for name,(lo,hi) in intervals.items():counts[name]+=int(np.count_nonzero((samples>=lo)&(samples<=hi)))
  for name,(lo,hi) in intervals.items():
   p=max(0.,min(upper,hi)-max(0.,lo))/upper;k=counts[name]
   trials.append({'null':'uniform_alpha','upper':upper,'threshold_name':name,'T_threshold':tolerances[name],
    'samples':NMC,'seed':SEED+j,'hits':k,'estimated_probability':k/NMC,'MC_p_plus_one':(k+1)/(NMC+1),
    'Clopper_Pearson_95_interval':cp(k,NMC),'analytic_probability':p,'acceptance_alpha_interval':[lo,hi]})
 # A marginal experimental check needs no unavailable covariance of three ratios.
 z=float((predicted[0]-refs[0])/4.6e-6);ng=1_000_000;rng=np.random.Generator(np.random.PCG64(SEED+2))
 hits=int(np.count_nonzero(np.abs(rng.standard_normal(ng))>=abs(z)))
 marginal={'statistic_z_mu_e':z,'samples':ng,'seed':SEED+2,'hits':hits,
  'MC_p_plus_one':(hits+1)/(ng+1),'one_sided_95_probability_upper':float(-np.expm1(np.log(.05)/ng)),
  'analytic_log10_two_sided_p':float((np.log(2)+log_ndtr(-abs(z)))/np.log(10)),
  'model':'Gaussian quoted marginal uncertainty; zero reduced-to-physical matching error assumed for this diagnostic.',
  'verdict':'Experimental-precision equality is rejected for the reduced mu/e coordinate; full physical matching remains open.'}
 sigma=np.array([4.6e-6,.09/105.6583755,.09/.51099895069])
 low=np.log(refs-sigma)/powers;high=np.log(refs+sigma)/powers
 experiments={'alpha_inferred_separately':list(np.log(refs)/powers),'alpha_minus_each_inferred':list(alpha-np.log(refs)/powers),
  'one_standard_uncertainty_alpha_intervals':list(map(list,zip(low,high))),
  'simultaneous_one_sigma_intersection':'empty' if max(low)>min(high) else [float(max(low)),float(min(high))],
  'simultaneous_01_percent_verdict':'passes' if T<=.001 else 'does not pass'}
 # Sensitivity: spectral linear response, independent central difference, and rational bounds.
 calc=[rotor(32,N) for N in [24,32,48,64]];r=calc[-1];h=1e-3
 fd=(rotor(32+h)['alpha']-rotor(32-h)['alpha'])/(2*h)
 roots=[brentq(lambda A:rotor(A)['alpha']-target,30,34,xtol=2e-12) for target in intervals['one_per_mille']]
 rational=[];hh=F(1,100000)
 for A in [F(32)-hh,F(32),F(32)+hh]:rational.append(rational_rotor(A))
 # H-A I=D-A C, ||C||<=1. W''=-E''' and |E'''|<=12/g^2.
 g=min(F(x[0]['gap_lower']) for x in rational)-2*hh
 slope=(rational[2][1]-rational[0][1])/(2*hh)
 slope=I(slope.a-6*hh/g**2,slope.b+6*hh/g**2)
 wi=rational[1][1];af=(I(*prior['bcc_return_interval'])*cos_small(PI/63))*wi.sqrt()
 derivative=(1+af.power(2)/(32*(1-af.power(3))))*af*slope/(2*wi)
 brackets=[]
 exact_refs=[F('206.7682827'),F('1776.93')/F('105.6583755'),F('1776.93')/F('.51099895069')]
 lower_targets=[rational_log(x*F(999,1000))/int(k) for x,k in zip(exact_refs,powers)]
 upper_targets=[rational_log(x*F(1001,1000))/int(k) for x,k in zip(exact_refs,powers)]
 exact_targets=[I(max(x.a for x in lower_targets),max(x.b for x in lower_targets)),I(min(x.a for x in upper_targets),min(x.b for x in upper_targets))]
 for target,root,ti in zip(intervals['one_per_mille'],roots,exact_targets):
  lo=F(str(np.floor(root*1e7)/1e7));hi=lo+F(1,10000000)
  left=rational_rotor(lo);right=rational_rotor(hi)
  assert left[2].b<ti.a<=ti.b<right[2].a
  brackets.append({'A_bracket':[str(lo),str(hi)],'target_alpha':target,'rational_target_interval':ti.pair(),'left_certificate':left[0],'right_certificate':right[0]})
 eta=[roots[0]/32-1,roots[1]/32-1];tuning=1/min(abs(eta[0]),abs(eta[1]))
 sensitivity={'at_32':r,'cutoff_comparison':calc,'central_difference_step':h,'central_difference':fd,
  'derivative_crosscheck_difference':abs(fd-r['alpha_derivative']),'rigorous_dalpha_dA_interval':derivative.pair(),
  'derivative_rational_certificates':[x[0] for x in rational],'accepted_A_roots':roots,'accepted_A_certificates':brackets,
  'relative_A_variation':eta,'inverse_nearest_fractional_margin':tuning,
  'logarithmic_ratio_sensitivities':list(powers*32*r['alpha_derivative']),
  'status':'A=32 is imposed by the C8 normalization in this reduced family; persistence under full physical matching remains open.'}
 e_math=31+mathieu_b(2,-64)/4
 dh=1e-3
 db=(mathieu_b(2,-64-2*dh)-8*mathieu_b(2,-64-dh)+8*mathieu_b(2,-64+dh)-mathieu_b(2,-64+2*dh))/(12*dh)
 special={'expression':'E0(A)=A-1+b_2(-2A)/4; W(A)=b_2_prime(-2A)/2; a=Gamma(1/4)^4 sqrt(W) cos(pi/63)/(32 pi^3); alpha=a-log(1-a^3)/96.',
  'alpha_interval':prior['rotor_cutoff_stability'][-1]['alpha_interval'],'Mathieu_ground_energy':float(e_math),
  'Mathieu_energy_difference':abs(float(e_math)-r['energy']),'Mathieu_W_derivative_check':float(db/2),
  'Mathieu_W_difference':abs(float(db/2)-r['W']),
  'status':'Derived special-function representation. No reduction to elementary constants is established; no nonexistence theorem is claimed.',
  'measurement':'alpha_conn is a reduced coefficient, not an independently measured coupling.'}
 mq={'u':2.16,'d':4.70,'s':93.5};dq={'u':.07,'d':.07,'s':.8}
 predicted_sd=float(predicted[0]);experimental_sd=mq['s']/mq['d']
 quark={'C2_fund':'4/3','c3':3,'Pq_for_Delta_5_3_1':[59,25,7],
  'formula':'P_q(Delta)=2 Delta^2+Delta+4 c3/3; w_q=exp(-alpha P_q).',
  'same_charge_generation_ratio_invariance':'The common Casimir term cancels for every c3.',
  'down_sector_s_over_d_reduced':predicted_sd,'PDG2025_s_over_d_central':experimental_sd,
  'PDG2025_s_over_d_individual_90_percent_box':[(mq['s']-dq['s'])/(mq['d']+dq['d']),(mq['s']+dq['s'])/(mq['d']-dq['d'])],
  'relative_difference':predicted_sd/experimental_sd-1,
  'reference_scheme':'MSbar, common scale 2 GeV; individual quoted 90% intervals give an envelope, not a joint confidence interval.',
  'same_pattern_up_c_u_t_c_t_u':list(predicted),'same_pattern_down_s_d_b_s_b_d':list(predicted),
  'verdict':'The additive-colour extension does not reproduce the same-scale s/d reference. The microscopic quark derivation and its renormalization matching remain open.',
  'axiom_scope':'A1 supplies the minimal boundary, not the colour Casimir coefficient; c3=3 uses the later trace normalization and minimal-flux prescription.'}
 result={'status':'EXECUTED_REDUCED_MODEL_DIAGNOSTICS','program_sha256':sha(Path(__file__)),
  'input_sha256':{'reproducibility/results/quantitative_comparison/QUANTITATIVE_COMPARISON.json':sha(ROOT/'reproducibility/results/quantitative_comparison/QUANTITATIVE_COMPARISON.json')},
  'environment':{'Python':platform.python_version(),'NumPy':np.__version__,'SciPy':scipy.__version__},
  'alpha':alpha,'observed_relative_residuals':list(predicted/refs-1),'test_statistic_T':T,'Monte_Carlo':trials,
  'marginal_experimental_test':marginal,'experimental_precision':experiments,'sensitivity':sensitivity,'special_function':special,'quarks':quark,
  'sources':{'CODATA2022':'https://physics.nist.gov/cuu/Constants/Table/allascii.txt','PDG2025tau':'https://pdg.lbl.gov/2025/listings/rpp2025-list-tau.pdf',
   'PDG2025light':'https://pdg.lbl.gov/2025/listings/rpp2025-list-light-quarks.pdf','Mathieu':'https://dlmf.nist.gov/28.2'},
  'null_scope':'Uniform alpha laws are declared diagnostics, not consequences of QICT. No trials factor for prior model selection is estimated.'}
 (OUT/'REDUCED_HIERARCHY_TESTS.json').write_text(json.dumps(result,indent=2)+'\n')
 summary={k:v for k,v in result.items() if k not in ['sensitivity','special_function']}
 summary['sensitivity']={k:v for k,v in sensitivity.items() if k not in ['derivative_rational_certificates','accepted_A_certificates','cutoff_comparison']}
 summary['special_function']=special
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
