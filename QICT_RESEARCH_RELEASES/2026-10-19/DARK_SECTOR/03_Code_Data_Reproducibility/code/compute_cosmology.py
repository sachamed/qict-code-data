"""Reproducible standard-EFT/Planck-normalized illustrations, not QICT blind predictions.
Sources: Planck 2018 VI, LZ 2025, SIS and BBKS textbook approximations.
All numerical inputs selected independently of QICT spectral sector.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad, solve_ivp, simpson
from scipy.constants import c, G, hbar, pi, physical_constants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style
from numerical_methods import correlation_sine_transform
root=Path(__file__).parent
fig=root.parent/'../01_Main_Manuscript'/'figures'
if not fig.is_dir(): fig=root/'figures'
fig.mkdir(exist_ok=True,parents=True)
MPC=3.085677581491367e22
MSUN=1.98847e30
GEV_J=physical_constants['electron volt-joule relationship'][0]*1e9
GEV2_CM2=0.389379338e-27
G_MPC=G*MSUN/MPC/1e6  # Mpc (km/s)^2 / Msun, from the same SI constants
h=0.674; H0=100*h; Om=.315; Ob=.02237/h**2; Ocdm=Om-Ob; N_eff=3.046
omega_gamma=2.469e-5 /h**2; Or=omega_gamma*(1+0.22710731766*N_eff); Ol=1-Om-Or
sigma8=.811; ns=.965; zstar=1089.92; zdrag=1059.
def E(z):return np.sqrt(Om*(1+z)**3+Or*(1+z)**4+Ol)
def chi(z):return (c/1000)/H0*quad(lambda zz:1/E(zz),0,z,epsrel=1e-9)[0]
def DA(z):return chi(z)/(1+z)
def DAls(zl,zs):return (chi(zs)-chi(zl))/(1+zs)
def sound_speed(a):return (c/1000)/np.sqrt(3*(1+3*Ob*a/(4*omega_gamma)))
def rs(z):
 aend=1/(1+z)
 return quad(lambda a: sound_speed(a)/(a*a*H0*np.sqrt(Om/a**3+Or/a**4+Ol)),1e-12,aend,epsrel=1e-8)[0]
H_SI=(H0*1000/MPC)
rho_crit=3*H_SI**2/(8*pi*G)
rho_lam_Jm3=rho_crit*Ol*c*c
lam=3*Ol*H_SI**2/c**2
srho= np.sqrt((2*(1-Om)/Ol*.5/67.4)**2 + (.007/Ol)**2) # illustrative uncorrelated 1-sigma
# GeV^4 natural conversion: rho[J/m^3]* (hbar*c)^3/(GeV[J])^4
rho_GeV4=rho_lam_Jm3*(hbar*c)**3/GEV_J**4
r_s=rs(zstar); D_M=chi(zstar); theta=r_s/D_M; lA=pi/theta
# growth ODE in ln a: matter-clustering subhorizon, dark energy smooth, neglect rad pert.
def growth_rhs(x,y):
 a=np.exp(x);ez2=Om/a**3+Or/a**4+Ol
 dlnH=(-3*Om/a**3-4*Or/a**4)/(2*ez2)
 om_a=(Om/a**3)/ez2
 return [y[1], -(2+dlnH)*y[1] + 1.5*om_a*y[0]]
a_ini=.01;x_ini=np.log(a_ini)
sol=solve_ivp(growth_rhs,(x_ini,0),[a_ini,a_ini],dense_output=True,rtol=1e-10,atol=1e-12)
D0=sol.sol(0)[0]
def Df(z):
 y=sol.sol(-np.log1p(z));return (y[0]/D0,y[1]/y[0])
zs=np.linspace(0,5,301)
Ds=np.array([Df(z)[0] for z in zs]); fs8=np.array([Df(z)[0]*Df(z)[1]*sigma8 for z in zs])
# BBKS smooth-transfer shape, Sugiyama baryonic correction; not BAO transfer.
shape=Om*h*h*np.exp(-Ob*(1+np.sqrt(2*h)/Om))
def transfer(k):
 q=np.asarray(k)/shape
 return np.log1p(2.34*q)/(2.34*q)*(1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-.25)
k=np.logspace(-4,2.5,2500)
p0=k**ns*transfer(k)**2
rho_m=(rho_crit*Om)/MSUN*MPC**3 # Msun / Mpc^3

def top_hat(x):
 x=np.asarray(x);res=np.ones_like(x)
 w=np.abs(x)>1e-2
 res[w]=3*(np.sin(x[w])-x[w]*np.cos(x[w]))/x[w]**3
 res[~w]=1-x[~w]**2/10+x[~w]**4/280
 return res
def sig2(R,P):
 kw=k*R
 return simpson(k**3*P*top_hat(kw)**2/(2*pi*pi),x=np.log(k))
R8=8/h
A=sigma8**2/sig2(R8,p0)
P=A*p0
# mass function Press-Schechter with numerical log derivative
mass=np.logspace(10,15.5,500)
RM=(3*mass/(4*pi*rho_m))**(1/3)
sig=np.array([np.sqrt(sig2(r,P)) for r in RM])
dlnsig=np.gradient(np.log(sig),np.log(mass))
deltac=1.686
nu=deltac/sig
# dn/dlnM: sqrt(2/pi) rho/M nu |dlnsig/dlnM| exp(-nu^2/2)
dndlnM=np.sqrt(2/pi)*rho_m/mass*nu*np.abs(dlnsig)*np.exp(-nu**2/2)
# galaxy mass function constant illustrative efficiency epsilon
eta_star=.05
mstar=eta_star*Ob/Om*mass
ngal_log10=dndlnM*np.log(10)
# SIS lens geometry
zl=.3;zsour=1.;sigma_v=220.; Dl=DA(zl); Ds_l=DA(zsour); Dls=DAls(zl,zsour)
theta_E=4*pi*(sigma_v/(c/1000))**2 *Dls/Ds_l
arcsec=180/pi*3600
theta_E_arc=theta_E*arcsec
beta=0.5*theta_E
F=(1+zl)*(Dl*MPC)*(Ds_l*MPC)/(c*(Dls*MPC))
delay_sec=2*F*theta_E*beta
# NFW projected halo illustration; no particle realization
M200=1e14; c200=5.0
rho_crit_z =rho_crit*E(zl)**2/MSUN*MPC**3
r200=(3*M200/(4*pi*200*rho_crit_z))**(1/3)
rs_halo=r200/c200
fnfw=np.log1p(c200)-c200/(1+c200)
rho_s=M200/(4*pi*rs_halo**3*fnfw)
Sigma_crit=(c/1000)**2/(4*pi*G_MPC)*Ds_l/(Dl*Dls) # Msun/Mpc^2

def fproj(x):
 x=np.asarray(x,dtype=float)
 o=np.empty_like(x)
 lt=x<1-1e-8;gt=x>1+1e-8;eq=~(lt|gt)
 if np.any(lt):
  t=x[lt];o[lt]=(1-2/np.sqrt(1-t*t)*np.arctanh(np.sqrt((1-t)/(1+t))))/(t*t-1)
 if np.any(gt):
  t=x[gt];o[gt]=(1-2/np.sqrt(t*t-1)*np.arctan(np.sqrt((t-1)/(1+t))))/(t*t-1)
 o[eq]=1/3
 return o
# evaluate projected convergence and aperture shear
radii=np.logspace(-2,0.45,150)
Sig=2*rho_s*rs_halo*fproj(radii/rs_halo)
kap=Sig/Sigma_crit
kapbar=np.array([2*quad(lambda rr:float(2*rho_s*rs_halo*fproj([max(rr/rs_halo,1e-10)])[0]/Sigma_crit)*rr, 1e-7,R,epsrel=1e-5)[0]/R**2 for R in radii])
gammat=kapbar-kap
# production and scalar portal benchmarks
mH=125.1;mN=.939;fnuc=.30
mWimp1=40.;portal1=.01
mu1=mWimp1*mN/(mWimp1+mN)
sigma_SI1=(portal1**2*fnuc**2*mu1**2*mN**2/(4*pi*mH**4*mWimp1**2))*GEV2_CM2
mWimp2=300.;portal2=.20
mu2=mWimp2*mN/(mWimp2+mN)
sigma_SI2=(portal2**2*fnuc**2*mu2**2*mN**2/(4*pi*mH**4*mWimp2**2))*GEV2_CM2
# hh contact term ONLY: not full scalar Higgs-portal annihilation
sigma_v_hh=portal2**2/(64*pi*mWimp2**2)*np.sqrt(1-(mH/mWimp2)**2)*GEV2_CM2*c*100
omega_contact=.12*2.2e-26/sigma_v_hh
# Freeze-in via A -> S phi, exactly one dark daughter; gA=1, constant g*.
Mpl=1.22089e19;M_A=1000.;width=1e-20;mFIMP=10.;gstar=106.75
Yinf=135*width*Mpl/(8*pi**3*1.66*gstar*np.sqrt(gstar)*M_A**2)
omegaFI=2.742e8*mFIMP*Yinf
H_MA=1.66*np.sqrt(gstar)*M_A**2/Mpl
# Visualizations
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'figure.dpi':160})
plt.figure(figsize=(5.1,3.4));plt.plot(zs,Ds);plt.xlabel('Redshift z');plt.ylabel('D(z)/D(0)');plt.title('Linear growth in a Planck-normalized reference cosmology');plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'growth_reference.pdf');plt.close()
plt.figure(figsize=(5.1,3.4));plt.loglog(k,P);plt.xlabel('k [Mpc$^{-1}$]');plt.ylabel('P(k) [Mpc$^3$]');plt.title('BBKS smooth transfer, normalized to $\\sigma_8$');plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'matter_power_reference.pdf');plt.close()
plt.figure(figsize=(5.1,3.4));plt.loglog(mass,dndlnM);plt.xlabel('Halo mass $M$ [$M_\\odot$]');plt.ylabel('$dn/d\\ln M$ [Mpc$^{-3}$]');plt.title('Press--Schechter reference mass function, z=0');plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'halo_mass_function.pdf');plt.close()
plt.figure(figsize=(5.1,3.4));plt.loglog(radii,kap,label='$\\kappa(R)$');plt.loglog(radii,np.abs(gammat),label='$|\\gamma_t(R)|$');plt.xlabel('Projected radius [Mpc]');plt.ylabel('Dimensionless lensing response');plt.title('Illustrative NFW cluster ($M_{200}=10^{14}M_\\odot$)');plt.legend();plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'cluster_lensing_reference.pdf');plt.close()
xg=np.linspace(-1.25,1.25,160);yg=np.linspace(-1.25,1.25,160)
X,Y=np.meshgrid(xg,yg);RR=np.maximum(np.hypot(X,Y),0.006)
MAP=2*rho_s*rs_halo*fproj(RR/rs_halo)/Sigma_crit
plt.figure(figsize=(4.3,3.7));plt.imshow(np.log10(MAP), origin='lower',extent=[xg.min(),xg.max(),yg.min(),yg.max()],cmap='viridis',vmin=-2,vmax=.5);plt.colorbar(label='$\\log_{10}\\kappa$');plt.xlabel('x [Mpc]');plt.ylabel('y [Mpc]');plt.title('Analytic NFW convergence map');plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'cluster_kappa_map.pdf');plt.close()
# Extra sanity on growth and sigma8
out={
  'inputs':dict(h=h,H0_km_s_Mpc=H0,Omega_m=Om,Omega_b=Ob,Omega_cdm=Ocdm,Omega_r=Or,Omega_Lambda=Ol,sigma8=sigma8,ns=ns,zstar=zstar,z_lens=zl,z_source=zsour,epsilon_star_illustrative=eta_star),
  'cosmology':dict(rhoLambda_J_m3=rho_lam_Jm3,Lambda_m_minus2=lam,rhoLambda_GeV4=rho_GeV4,illustrative_68pct_input_propagation_fraction=srho,rs_zstar_Mpc=r_s,DM_zstar_Mpc=D_M,theta_star=theta,theta_star_100=theta*100,ell_acoustic=lA,growth_z0p5=Df(.5)[0],f_z0p5=Df(.5)[1],fsigma8_z0p5=Df(.5)[0]*Df(.5)[1]*sigma8),
  'lensing':dict(velocity_dispersion_km_s=sigma_v,thetaEinstein_arcsec=theta_E_arc,Dl_Mpc=Dl,Ds_Mpc=Ds_l,Dls_Mpc=Dls,source_beta_fraction_E=beta/theta_E,time_delay_days=delay_sec/86400,M200_Msun=M200,c200=c200,r200_Mpc=r200,NFW_kappa_at_0p1Mpc=float(np.interp(.1,radii,kap)),NFW_gammat_at_0p1Mpc=float(np.interp(.1,radii,gammat))),
  'halo':dict(rho_m_Msun_Mpc3=rho_m,bbks_gamma=shape,sigma8_reproduced=np.sqrt(sig2(R8,P)),dndlnM_1e12_Mpc3=float(np.interp(1e12,mass,dndlnM)),dndlnM_1e14_Mpc3=float(np.interp(1e14,mass,dndlnM))),
  'dark_matter_benchmarks':dict(m40_GeV=mWimp1,lambda40=portal1,SI40_cm2=sigma_SI1,relative_to_LZ_40GeV_min=sigma_SI1/(2.2e-48),m300_GeV=mWimp2,lambda300=portal2,SI300_cm2=sigma_SI2,hh_contact_only_cm3_s=sigma_v_hh,contact_only_relic_approx=omega_contact,freezein_MA_GeV=M_A,freezein_width_GeV=width,freezein_mchi_GeV=mFIMP,freezein_Y=Yinf,freezein_omega_h2=omegaFI,freezein_width_over_H_at_MA=width/H_MA)
}
(root/'results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
# Nonthermal freeze-in production history (numerical integration of decay reaction density)
from scipy.special import k1
xprod=np.logspace(-3,2,500)
pre=45*width*Mpl/(4*pi**4*1.66*gstar*np.sqrt(gstar)*M_A**2)
Yprod=pre*np.array([quad(lambda t:t**3*k1(t),xprod[0],x,epsabs=2e-12,epsrel=2e-12)[0] for x in xprod])
plt.figure(figsize=(5.1,3.4));plt.semilogx(xprod,Yprod/Yinf)
plt.xlabel('$x=M_A/T$');plt.ylabel('$Y_\\chi(x)/Y_\\chi(\\infty)$')
plt.title('Freeze-in from equilibrium parent decay: MB approximation')
plt.ylim(0,1.04);plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'freezein_yield.pdf');plt.close()
# Illustrative galaxy mass function obtained via a fixed stellar fraction
plt.figure(figsize=(5.1,3.4));plt.loglog(mstar,ngal_log10)
plt.xlabel('Illustrative galaxy stellar mass [$M_\\odot$]')
plt.ylabel('$dn/d\\log_{10} M_\\star$ [Mpc$^{-3}$]')
plt.title('Constant stellar-efficiency model')
plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'stellar_mass_function.pdf');plt.close()
# Cluster tangential critical radius for reference NFW mass=1e15 Msun, concentration 4
from scipy.optimize import brentq
mass_cluster=1e15; cc=4.
rr=(3*mass_cluster/(4*pi*200*rho_crit_z))**(1/3); rc=rr/cc
fcluster=np.log1p(cc)-cc/(1+cc)
rhos=mass_cluster/(4*pi*rc**3*fcluster)
def mean_cluster_kappa(R):
 integr=quad(lambda rad: 2*rhos*rc*fproj(np.array([max(rad/rc,1e-12)]))[0]/Sigma_crit * rad,1e-9,R,epsrel=1e-7)[0]
 return 2*integr/R**2
rng=np.logspace(-6,.4,50)
cvals=[mean_cluster_kappa(x) for x in rng]
roots=[]
for i in range(len(rng)-1):
 if (cvals[i]-1)*(cvals[i+1]-1)<0:
  roots.append(brentq(lambda r:mean_cluster_kappa(r)-1,rng[i],rng[i+1]))
crit=roots[0] if roots else None
extra={'freezein_relic_numerical_to_analytic_ratio':float(Yprod[-1]/Yinf),
       'cluster_M200_Msun':mass_cluster,'cluster_concentration':cc,
       'cluster_tangential_critical_R_Mpc':crit,
       'cluster_tangential_critical_theta_arcsec':(crit/Dl*arcsec) if crit else None,
       'galaxy_stellar_mass_at_Mhalo_1e12_Msun':float(eta_star*Ob/Om*1e12)}
res=json.loads((root/'results.json').read_text());res['additional']=extra
(root/'results.json').write_text(json.dumps(res,indent=2))
print('EXTRA',json.dumps(extra,indent=2))
# Nonlinear isolated spherical top-hat dynamics: dynamical halo-collapse proxy (NOT N-body).
a_i=.01;delta_i=.04
Mtop=1e14
Rinit=(3*Mtop/(4*pi*rho_m/a_i**3*(1+delta_i)))**(1/3)
Rprime=Rinit*(1-delta_i/(3*(1+delta_i)))
def tophat_rhs(x,y):
 a=np.exp(x);E2=Om/a**3+Or/a**4+Ol
 dh=(-3*Om/a**3-4*Or/a**4)/(2*E2)
 Hsq=H0**2*E2
 return [y[1],(-G_MPC*Mtop/(y[0]**2)-Or*H0**2/a**4*y[0]+Ol*H0**2*y[0])/Hsq-dh*y[1]]
def hit_collapse(x,y):return y[0]-.01
hit_collapse.terminal=True;hit_collapse.direction=-1
top=solve_ivp(tophat_rhs,(np.log(a_i),0),[Rinit,Rprime],events=hit_collapse,max_step=.005,rtol=1e-9,atol=1e-11)
ac=np.exp(top.t_events[0][0]) if len(top.t_events[0]) else None
Rbg=lambda a:a*(3*Mtop/(4*pi*rho_m))**(1/3)
plt.figure(figsize=(5.1,3.4));plt.plot(np.exp(top.t),top.y[0],label='Overdense spherical patch');
plt.plot(np.exp(top.t),Rbg(np.exp(top.t)),ls='--',label='Hubble-flow reference')
plt.xlabel('Scale factor a');plt.ylabel('Physical patch radius [Mpc]')
plt.title('Nonlinear spherical-collapse proxy; single uniform top-hat')
plt.legend();plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'spherical_collapse_reference.pdf');plt.close()
res=json.loads((root/'results.json').read_text());res['spherical_top_hat']={'mass_Msun':Mtop,'initial_a':a_i,'initial_overdensity':delta_i,'initial_radius_Mpc':Rinit,'collapse_definition_Mpc':.01,'collapse_scale_factor':float(ac) if ac is not None else None,'numerical_integration_success':bool(top.success)}
(root/'results.json').write_text(json.dumps(res,indent=2));print('COLLAPSE',res['spherical_top_hat'])
# Linear real-space matter/galaxy correlation in the smooth-power reference.
from scipy.integrate import simpson
rg=np.logspace(0.1,2.1,130)
# Oscillatory sine transform resolves kr oscillations independently of the log-k grid.
xi_pairs=[correlation_sine_transform(r,A,ns,transfer) for r in rg]
xik=np.array([v[0] for v in xi_pairs])
bgal=1.5
xig=bgal*bgal*xik
plt.figure(figsize=(5.1,3.4));plt.plot(rg,rg**2*xig,label='$b_g=1.5$ (illustrative)')
plt.axhline(0,color='black',lw=.6)
plt.xlabel('Comoving separation r [Mpc]');plt.ylabel('$r^2\\,\\xi_g(r)$ [Mpc$^2$]')
plt.title('Linear galaxy correlation from the smooth reference spectrum')
plt.grid(alpha=.23);plt.legend();plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'galaxy_correlation_reference.pdf');plt.close()
# Hydrogen Saha-equilibrium reference; NOT the freeze-out recombination history.
from scipy.constants import Boltzmann as kB, hbar as hb, m_e as me, m_p as mp, e as elementary_charge
from scipy.optimize import brentq
Yp=.245;T0=2.7255;chiH=13.5984*elementary_charge
nH0=(1-Yp)*rho_crit*Ob/mp
zrec=np.linspace(800,1900,600)
xerec=[]
for zz in zrec:
 T=T0*(1+zz);nH=nH0*(1+zz)**3
 Q=(me*kB*T/(2*pi*hb*hb))**1.5*np.exp(-chiH/(kB*T))/nH
 xe=(2*Q)/(np.sqrt(Q*Q+4*Q)+Q)
 xerec.append(xe)
xerec=np.array(xerec)
# locate x_e = 0.5 crossing, if available
zhalf=float(np.interp(.5,xerec,zrec))
plt.figure(figsize=(5.1,3.4));plt.plot(zrec,xerec,label='Hydrogen Saha equilibrium')
plt.axvline(zstar,color='gray',ls='--',label='Input $z_*=1089.92$')
plt.xlabel('Redshift z');plt.ylabel('Hydrogen ionized fraction $x_e$')
plt.title('Hydrogen ionization in Saha equilibrium')
plt.legend();plt.grid(alpha=.23);plt.tight_layout();plot_style.save_figure(plt.gcf(),fig/'saha_ionization_reference.pdf');plt.close()
res=json.loads((root/'results.json').read_text())
res['galaxy_correlation']={'illustrative_constant_bias':bgal,'xi_g_at_10Mpc':float(bgal*bgal*correlation_sine_transform(10,A,ns,transfer)[0]), 'xi_g_at_20Mpc':float(bgal*bgal*correlation_sine_transform(20,A,ns,transfer)[0]), 'quadrature_error_estimate_max':float(max(v[1] for v in xi_pairs)*bgal*bgal), 'method':'oscillatory sine transform on [0,infinity)'}
res['saha_recombination']={'assumed_helium_fraction_Yp':Yp,'CMB_temperature_K':T0,'equilibrium_xe_half_redshift':zhalf,'xe_equilibrium_at_zstar':float(np.interp(zstar,zrec,xerec))}
(root/'results.json').write_text(json.dumps(res,indent=2))
print('GAL / SAHA',res['galaxy_correlation'],res['saha_recombination'])
