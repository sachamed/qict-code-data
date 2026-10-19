"""Analytically specified reference tests for scale-independent phenomenological gravity
and a one-zone gas regulator; no QICT microphysical parameters were used.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp, simpson, cumulative_trapezoid
from scipy.constants import c
from scipy.interpolate import PchipInterpolator
P=Path(__file__).resolve().parents[1]
(P/'data').mkdir(exist_ok=True)
Om0=.315; h0=.674; Or0=(2.469e-5/h0**2)*(1+0.22710731766*3.046); Ol0=1-Om0-Or0; sigma80=.811
z=.5;a=np.exp(-np.log1p(z))

def grow(mu):
 def rhs(x,y):
  a=np.exp(x); E2=Om0/a**3+Or0/a**4+Ol0; oma=(Om0/a**3)/E2
  fr=2-(3*Om0/a**3+4*Or0/a**4)/(2*E2)
  return [y[1], 1.5*oma*mu*y[0]-fr*y[1]]
 ai=.01
 # growing adiabatic initial condition in matter era: D~a^p, p>0 of p^2+(1/2)p-(3/2)mu=0
 p=(-.5+np.sqrt(.25+6*mu))/2
 s=solve_ivp(rhs,(np.log(ai),0),[ai**p,p*ai**p],dense_output=True,rtol=2e-11,atol=1e-13)
 def v(z):
  y=s.sol(-np.log1p(z));return float(y[0]),float(y[1]/y[0])
 return v
base=grow(1.0);mod=grow(1.05)
D0base, f0base=base(0)
Dhalfbase,fhalfbase=base(.5)
D0mod,f0mod=mod(0)
Dhalfmod,fhalfmod=mod(.5)
# Normalize both with identical early primordial amplitude at a=0.01? Different initial exponents make
# these amplitudes incomparable; instead compare at a fixed reference value ai with same D(ai)=ai,
# and choose each matter-dominated logarithmic growing p. Base D(ai)=ai, modified D(ai)=ai^p.
# Rescale modified to same initial density amplitude at ai=0.01.
pmod=(-.5+np.sqrt(.25+6*1.05))/2
renorm=(.01)/(.01**pmod)
Dbhalf=Dhalfbase/D0base
Drelmod=Dhalfmod*renorm/D0base
Drelbase=Dhalfbase/D0base
Sigma=1.05*(1+.9)/2
EGbase=Om0/fhalfbase
EGmod=Om0*Sigma/fhalfmod
equal_time_power_ratio=Sigma**2*(Drelmod/Drelbase)**2
# Angular convergence is an integral over lens redshift, not the value at z=0.5.
def transfer(k):
 q=np.asarray(k)/(.315*h0*h0*np.exp(-(.02237/h0**2)*(1+np.sqrt(2*h0)/.315)))
 return np.log1p(2.34*q)/(2.34*q)*(1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-.25)
k=np.logspace(-5,2.9,5000); raw=k**.965*transfer(k)**2
window=lambda x:3*(np.sin(x)-x*np.cos(x))/x**3
norm=.811**2/simpson(k**3*raw*window(k*8/h0)**2/(2*np.pi**2),x=np.log(k))
interp=PchipInterpolator(np.log(k),np.log(norm*raw),extrapolate=False)
zg=np.linspace(0,1,4001); ez=np.sqrt(Om0*(1+zg)**3+Or0*(1+zg)**4+Ol0)
chi=cumulative_trapezoid(c/1000/(100*h0)/ez,zg,initial=0)
zz=zg[1:];cc=chi[1:]
db=np.array([base(z)[0]/D0base for z in zz])
dm=np.array([mod(z)[0]*renorm/D0base for z in zz])
geom=(c/1000/(100*h0)/ez[1:])*(1+zz)**2*((chi[-1]-cc)/chi[-1])**2
angular_ratios={}
for ell in (50,200,1000):
 ke=(ell+.5)/cc; pk=np.zeros_like(ke); inside=(ke>=k[0])&(ke<=k[-1])
 pk[inside]=np.exp(interp(np.log(ke[inside])))
 angular_ratios[str(ell)]=float(Sigma**2*simpson(geom*pk*dm**2,x=zz)/simpson(geom*pk*db**2,x=zz))
# Simple closed gas regulator: constant halo growth and constant tau_star, return fraction and loading
fb=.02237/(h0**2*Om0)
Mdot_h=50. # Msun/yr
Mdot_in=fb*Mdot_h
returnfrac=.40;mass_loading=1.0;star_eff=.05;tdyn_Gyr=.1;t_Gyr=3.
lam=(1-returnfrac+mass_loading)*star_eff/tdyn_Gyr # Gyr^-1
inflow_per_Gyr=Mdot_in*1e9
Mg=(inflow_per_Gyr/lam)*(1-np.exp(-lam*t_Gyr))
sfr=star_eff*Mg/(tdyn_Gyr*1e9)
Mstar=(1-returnfrac)*inflow_per_Gyr/(1-returnfrac+mass_loading)*(t_Gyr-(1-np.exp(-lam*t_Gyr))/lam)
expectedstar=(1-returnfrac)*Mdot_in*1e9*t_Gyr
assert 0<Mstar<expectedstar
result={
 'background':{'Omega_m0':Om0, 'model_A_mu':1.0,'model_A_eta':1.0,'model_B_mu':1.05,'model_B_eta':.9,'normalization_a':.01},
 'growth':{'f_z0p5_GR':fhalfbase,'f_z0p5_B':fhalfmod,'EG_z0p5_GR':EGbase,'EG_z0p5_B':EGmod,
 'Sigma_B':Sigma, 'equal_time_Weyl_power_B_over_GR_z0p5':equal_time_power_ratio,'angular_convergence_B_over_GR_equal_initial_density':angular_ratios,
 'GR_Dhalf_normalized_z0':Dbhalf,'B_Dhalf_with_same_D_at_ai_relative_GR_z0':Drelmod},
 'star_formation':{'baryon_fraction':fb,'halo_accretion_Msun_per_yr':Mdot_h,'gas_inflow_Msun_per_yr':Mdot_in,
 'SFR_Msun_per_yr_at_3_Gyr':sfr,'gas_mass_Msun_at_3_Gyr':Mg,'stellar_mass_Msun_at_3_Gyr':Mstar,
 'epsilon_per_dynamical_time':star_eff,'recycling_fraction':returnfrac,'wind_mass_loading':mass_loading,
 'dynamical_time_Gyr':tdyn_Gyr,'time_Gyr':t_Gyr}}
(P/'data'/'metric_growth_galaxy.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
