"""Auditable cosmological consistency computations for the QICT companion article.

All dark-particle parameters and Planck background are independently chosen
benchmark inputs, *not* derived QICT predictions. This module computes a
controlled toy constant-s-wave freeze-out and linear weak lensing spectra.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.interpolate import PchipInterpolator
from scipy.special import kn
from scipy.constants import c, pi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style
from numerical_methods import constant_rate_freezeout

ROOT = Path(__file__).parent
FIG = ROOT.parent/'../01_Main_Manuscript'/'figures'
if not FIG.is_dir(): FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
prior=json.loads((ROOT/'results.json').read_text())
inputs=prior['inputs']; dmb=prior['dark_matter_benchmarks']
h=inputs['h']; H0=inputs['H0_km_s_Mpc']; Om=inputs['Omega_m']; Or=inputs['Omega_r']; Ol=inputs['Omega_Lambda']; ns=inputs['ns'];sig8=inputs['sigma8']
GEV2_CM2=0.389379338e-27
MPL=1.22089e19
OMEGA_CONV=2.742e8
CM_PER_SECOND = c*100

# Spin-independent Higgs-portal LZ 2024+2022 4.2 tonne-year combined result;
# 2.2e-48 cm2 is quoted at a mass of 40 GeV/c2, not a universal limit curve.
LZ_LIM40=2.2e-48
sig40=dmb['SI40_cm2']
lam40=dmb['lambda40']
lam40_max=lam40*np.sqrt(LZ_LIM40/sig40)
rejection=sig40/LZ_LIM40

# Exactly one daughter per A decay in the Maxwell-Boltzmann freeze-in limit;
# inversion is deliberately an observational fit (NOT QICT prediction).
omega_planck_cdm_h2=.120
width_given=dmb['freezein_width_GeV']
width_required=width_given*omega_planck_cdm_h2/dmb['freezein_omega_h2']
ratio_required=dmb['freezein_width_over_H_at_MA']*width_required/width_given

# Numerical constant-s-wave thermal freeze-out, specified independently of QICT.
# Scalar species gchi=1, equilibrium distributions MB, constant entropy+energy g*.
# Equation: dY/dx= -[s/(Hx)] <sigma v> (Y**2-Yeq**2), x=m/T.
mchi=300.;gchi=1.; gstar=90.0
svcm3=dmb['hh_contact_only_cm3_s']
svgev2=svcm3/(GEV2_CM2*CM_PER_SECOND)
coef=(2*pi*pi/45)/(1.66)*gstar/(gstar**.5)*MPL*mchi*svgev2
# Yeq=45/(4pi^4)*g/g_s x^2 K_2(x)
def Yeq(x):return (45/(4*pi**4))*gchi/gstar * x*x*kn(2,x)
# in t=log x coordinate, dy/dt = - coef / x * (Y^2 - Yeq^2)
# numerical integration in original Y with high-precision stiff solver.
x0=8.; xmax=4000.; y0=Yeq(x0)
def fun(t,y):
 x=np.exp(t)
 return [-coef/x*(y[0]*y[0]-Yeq(x)**2)]
fo=constant_rate_freezeout(coef, gstar=gstar, mass=mchi, x0=x0, xf=xmax)
sol=fo['solution']
yinf=fo['Y_infinity']; omega_fo=OMEGA_CONV*mchi*yinf
xx=np.logspace(np.log10(x0),np.log10(600),400)
yfo=sol.sol(np.log(xx))[0]/fo['scale']
yf_eq=np.array([Yeq(x) for x in xx])
# kinetic decoupling, coannihilation, resonance, running g*, entropy production etc omitted.
plt.figure(figsize=(5.1,3.4));plt.loglog(xx,yfo,label='Numerical constant-s-wave freeze-out');plt.loglog(xx,yf_eq,ls='--',label='MB equilibrium');plt.xlabel('$x=m_\\chi/T$');plt.ylabel('$Y=n/s$');plt.legend();plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),FIG/'freezeout_toy_exact_ode.pdf');plt.close()

# Shear and convergence angular auto spectrum for a SINGLE SOURCE PLANE at z=1,
# flat standard GR baseline, Limber+Born and linear BBKS normalized by observed sig8.
MPC_M=3.085677581491367e22
vlight=(c/1000)
omega_b=inputs['Omega_b'];
shape=Om*h*h*np.exp(-omega_b*(1+np.sqrt(2*h)/Om))
def TBBKS(k):
 q=k/shape
 return np.log1p(2.34*q)/(2.34*q)*(1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-.25)
k=np.logspace(-5,2.9,5000)
Pshape=k**ns*TBBKS(k)**2
def Wh(x):
 x=np.array(x); out=np.ones_like(x)
 v=np.abs(x)>1e-2
 out[v]=3*(np.sin(x[v])-x[v]*np.cos(x[v]))/x[v]**3
 out[~v]=1-x[~v]**2/10+x[~v]**4/280
 return out
R8=8/h
sigma8_sqr_base=simpson(k**3*Pshape*Wh(k*R8)**2/(2*pi*pi),x=np.log(k))
Anorm=sig8**2/sigma8_sqr_base
P0=Anorm*Pshape
interpP=PchipInterpolator(np.log(k),np.log(P0),extrapolate=False)
def E(z):return np.sqrt(Om*(1+z)**3+Or*(1+z)**4+Ol)
# comoving distance cumulative integral over z
from scipy.integrate import cumulative_trapezoid
zgrid=np.linspace(0,1,4001)
chigrid=cumulative_trapezoid(vlight/H0/E(zgrid),zgrid,initial=0)
chi_s=chigrid[-1]
# integrate linear growth ODE with matter-era initial condition a=.01
xini=np.log(.01)
def rhs(x,y):
 a=np.exp(x); e2=Om/a**3+Or/a**4+Ol
 dl=(-3*Om/a**3-4*Or/a**4)/(2*e2)
 return [y[1],-(2+dl)*y[1]+1.5*(Om/a**3)/e2*y[0]]
growth=solve_ivp(rhs,(xini,0),[.01,.01],dense_output=True,rtol=2e-10,atol=1e-12)
if not growth.success:raise RuntimeError('growth failure')
Ds=growth.sol(-np.log1p(zgrid))[0]/growth.sol(0)[0]
# lens Limber C_l = int dchi W(chi)^2/chi^2 P((ell+1/2)/chi,z)
# standard kernel W=3 Omega_m H0^2/(2 c^2)*chi/a*(chi_s-chi)/chi_s.
z=zgrid[1:];chi=chigrid[1:];D=Ds[1:]
pre=(1.5*Om*(H0/vlight)**2)**2
kernel=pre*(1+z)**2*((chi_s-chi)/chi_s)**2*D**2
jac=vlight/H0/E(z)
ells=np.unique(np.r_[np.rint(np.geomspace(10,1500,160)).astype(int),50,200,1000])
Cl=[]
for ell in ells:
 ke=(ell+.5)/chi
 P=np.zeros_like(ke)
 inside=(ke>k[0]) & (ke<k[-1]);P[inside]=np.exp(interpP(np.log(ke[inside])))
 Cl.append(float(simpson(jac*kernel*P,x=z)))
Cl=np.array(Cl)
if np.any(Cl<0) or np.any(~np.isfinite(Cl)):
 raise RuntimeError('invalid convergence spectrum')
plt.figure(figsize=(5.1,3.4));plt.loglog(ells,ells*(ells+1)*Cl/(2*pi));plt.xlabel('Multipole $\\ell$');plt.ylabel('$\\ell(\\ell+1) C_\\ell^{\\kappa\\kappa}/(2\\pi)$');plt.title('Single-plane linear convergence: BBKS/GR reference');plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),FIG/'linear_cosmic_shear_limber.pdf');plt.close()

results={
 'note':'All models are conventional selected effective benchmarks, not derived QICT observables.',
 'direct_detection':{'LZ_4p2ty_40_GeV_SI_90pct_cm2':LZ_LIM40,'benchmark_cm2':sig40,'benchmark_over_limit':rejection,'portal_coupling_max_given_model':lam40_max,'fN_input':0.3},
 'freezein':{'width_for_observed_Omegac_h2_GeV':width_required,'Omega_target_from_Planck':omega_planck_cdm_h2,'width_over_H_at_T_MA':ratio_required,'calibration_not_prediction':True},
 'freezeout':{'mchi_GeV':mchi,'gstar_constant':gstar,'swave_cm3_s':svcm3,'swave_GeV_minus2':svgev2,'x_ini':x0,'Yeq_at_initial':y0,'Y_endpoint':fo['Y_endpoint'],'Y_final':yinf,'omega_h2':omega_fo,'x_endpoint':xmax,'post_endpoint_tail_fraction':fo['tail_fraction_of_endpoint'],'asymptotic_tail_included':True,'solver': 'scipy Radau', 'success':bool(sol.success), 'note':'contact hh-only rate used as CONSTANT annihilation throughout: pedagogical closure only; NOT a Higgs portal freezeout computation'},
 'weak_lensing':{'source_z':1.0,'chi_source_Mpc':float(chi_s),'ell_50_Cl':float(np.interp(50,ells,Cl)),'ell_200_Cl':float(np.interp(200,ells,Cl)),'ell_1000_Cl':float(np.interp(1000,ells,Cl)),'ell_grid':ells.tolist(),'cl_kappa_grid':Cl.tolist(),'assumptions':'linear BBKS, smooth GR, Planck sigma8 normalization, Born+Limber, single source plane, no nonlinear correction'},
 'lensing_formula_correction':'Deflection prefactor for potentials Phi,Psi is 1/c^2, not 2/c^2; GR Phi=Psi gives 2/c^2 Newtonian potential integral.',
}
(ROOT/'closure_results.json').write_text(json.dumps(results,indent=2))
print(json.dumps({key:val for key,val in results.items() if key!='weak_lensing'},indent=2))
print('LENS',[(v,float(np.interp(v,ells,Cl))) for v in (50,200,1000)])
