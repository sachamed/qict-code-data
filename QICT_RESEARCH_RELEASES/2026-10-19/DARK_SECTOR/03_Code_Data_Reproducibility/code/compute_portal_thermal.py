"""Thermal tree-level two-body Higgs-portal benchmark in the stated convention.

All open hh, WW, ZZ and charged-fermion channels, including hh interference,
are included. Fixed vacuum masses/couplings and g*=90 define this benchmark;
loop final states, finite-temperature self-energies and running are separate.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.special import kve, kn, roots_genlaguerre, roots_legendre
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style

ROOT=Path(__file__).resolve().parent
M=300.; LAMBDA=.20; V=246.22; MH=125.1; WIDTH_H=.00407
MW=80.379; MZ=91.1876; GSTAR=90.; MPL=1.22089e19
FERMIONS=[('e',.00051099895,1),('mu',.1056583755,1),('tau',1.77686,1),
          ('u',.00216,3),('d',.00467,3),('s',.093,3),
          ('c',1.27,3),('b',4.18,3),('t',172.76,3)]
COSTH, ANG_WEIGHTS=roots_legendre(48)
KIN, KIN_WEIGHTS=roots_genlaguerre(48,.5)


def rates(s):
    """CM Moller sigma*v=2*beta_initial*sigma, in GeV^-2."""
    s=np.atleast_1d(s).astype(float)
    energy=np.sqrt(s); den=(s-MH**2)**2+(MH*WIDTH_H)**2
    width=np.zeros_like(s)
    for name,mf,nc in FERMIONS:
        width+=nc*mf**2*energy/(8*np.pi*V**2)*np.maximum(1-4*mf**2/s,0)**1.5
    for mv,factor in [(MW,16),(MZ,32)]:
        q=mv**2/s
        width+=energy**3/(factor*np.pi*V**2)*np.sqrt(np.maximum(1-4*q,0))*(1-4*q+12*q*q)
    visible=2*LAMBDA**2*V**2*width/(energy*den)
    bh=np.sqrt(np.maximum(1-4*MH**2/s,0))
    pi=np.sqrt(np.maximum(s/4-M**2,0));pf=np.sqrt(np.maximum(s/4-MH**2,0))
    # t-mS^2 = mh^2-s/2+2 p_i p_f cos(theta).
    t=MH**2-s[:,None]/2+2*pi[:,None]*pf[:,None]*COSTH
    u=MH**2-s[:,None]/2-2*pi[:,None]*pf[:,None]*COSTH
    amp=LAMBDA*(1+3*MH**2/(s[:,None]-MH**2+1j*MH*WIDTH_H)+LAMBDA*V**2*(1/t+1/u))
    hh=bh/(32*np.pi*s)*np.sum(np.abs(amp)**2*ANG_WEIGHTS,axis=1)
    return hh,visible


def average(x):
    """Gondolo--Gelmini integral, with scaled Bessel functions and Laguerre weights."""
    u=2+KIN/x
    hh,visible=rates(M*M*u*u)
    factor=u**3*np.sqrt((u*u-4)/KIN)*kve(1,x*u)/(8*kve(2,x)**2)
    return float(np.sum(KIN_WEIGHTS*factor*(hh+visible)))


def main():
    xx=np.geomspace(8,4000,180)
    thermal=np.array([average(x) for x in xx])
    rate=PchipInterpolator(np.log(xx),np.log(thermal))
    constant=(2*np.pi**2/45)/1.66*np.sqrt(GSTAR)*MPL*M
    eq=lambda x:45/(4*np.pi**4*GSTAR)*x*x*kn(2,x)
    scale=1e12
    def rhs(t,u):
        x=np.exp(t);sv=np.exp(rate(t))
        return [-constant*sv/x/scale*(u[0]**2-(scale*eq(x))**2)]
    sol=solve_ivp(rhs,(np.log(8),np.log(4000)),[eq(8)*scale],method='Radau',
                  rtol=2e-9,atol=1e-12,max_step=.03,dense_output=True)
    assert sol.success and sol.y[0,-1]>0
    # Integrate the tail by q=1/x, including the x dependence of the thermal rate.
    nodes,weights=roots_legendre(24);q=(nodes+1)/(2*4000)
    tail=constant*np.sum(weights*np.array([average(1/v) for v in q]))/(2*4000)
    endpoint=float(sol.y[0,-1]/scale)
    yi=1/(1/endpoint+tail)
    hh0,vis0=rates(4*M*M)
    conv=.389379338e-27*29979245800.
    result={'model':'Real Z2-odd scalar, -lambda_HS S^2 HdagH/2; vacuum tree-level 2-body benchmark',
            'inputs':{'mass_GeV':M,'lambda_HS':LAMBDA,'v_GeV':V,'mh_GeV':MH,'Gamma_h_GeV':WIDTH_H,
                      'mW_GeV':MW,'mZ_GeV':MZ,'gstar_constant':GSTAR,
                      'fermions_name_fixed_mass_GeV_colour':FERMIONS},
            'sigma_v_threshold_hh_cm3_s':float(hh0[0]*conv),
            'sigma_v_threshold_other_tree_cm3_s':float(vis0[0]*conv),
            'sigma_v_threshold_total_cm3_s':float((hh0[0]+vis0[0])*conv),
            'sigma_v_thermal_x20_cm3_s':average(20)*conv,
            'Y_at_x4000':endpoint,'Y_infinity':yi,'Omega_h2':2.742e8*M*yi,
            'tail_inverse_yield':float(tail),'x_grid':xx.tolist(),
            'thermal_sigma_v_GeV_minus2':thermal.tolist(),
            'scope':'Fixed vacuum parameters, Maxwell-Boltzmann kinetic equilibrium, constant gstar. QICT particle/vertices remain independent inputs.'}
    (ROOT.parent/'data/PORTAL_THERMAL_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(5.1,3.4))
    ax.semilogx(xx,thermal*conv,label='All tree-level two-body channels')
    contact=LAMBDA**2/(64*np.pi*M*M)*np.sqrt(1-MH**2/M**2)*conv
    ax.axhline(contact,color='#777777',ls='--',label='hh contact term')
    ax.set(xlabel=r'$x=m_S/T$ (log scale)',ylabel=r'$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]')
    ax.legend();fig.tight_layout();plot_style.save_figure(fig,ROOT.parent/'../01_Main_Manuscript/figures/portal_thermal_rates.pdf');plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['x_grid','thermal_sigma_v_GeV_minus2','inputs']},indent=2))


if __name__=='__main__':main()
