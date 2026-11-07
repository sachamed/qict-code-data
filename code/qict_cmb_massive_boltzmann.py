"""Single-massive-neutrino QICT one-pole CAMB spectrum.
The local scalar w(a) is reconstructed with a Fermi-Dirac massive species;
CAMB itself evolves the corresponding full massive-neutrino hierarchy.
"""
from pathlib import Path
import numpy as np
import json
import camb
from scipy.optimize import brentq
from numpy.polynomial.laguerre import laggauss

def history(n=2500, H0=69.28, ombh2=.02237, omch2=.119,
            nnu=3.044,mnu=.06,lam=.03776,qcrit=1e5):
    h=H0/100
    om=(ombh2+omch2)/h**2
    og=2.469e-5/h**2
    a=np.geomspace(1e-8,1,n)
    Tnu=2.7255*(4/11)**(1/3)*8.617333262145e-5
    q,w0=laggauss(96)
    weights=w0/(1+np.exp(-q))
    y=a[:,None]*(mnu/Tnu)
    en=np.sqrt(q[None,:]**2+y**2)
    I=np.sum(weights[None,:]*q[None,:]**2*en,axis=1)
    J=np.sum(weights[None,:]*q[None,:]**4/en,axis=1)
    I0=7*np.pi**4/120
    fac=og*a**-4*(15/np.pi**4)*(4/11)**(4/3)
    nm=nnu/3
    rho_nu=fac*((nnu-nm)*I0+nm*I)
    p_nu=fac*((nnu-nm)*I0/3+nm*J/3)
    rg=og*a**-4
    def f(E):
        z=E/qcrit; return lam*z/(1+z*z)
    def fp(E):
        z=E/qcrit; return lam/qcrit*(1-z*z)/(1+z*z)**2
    floor=1-om-og-rho_nu[-1]-f(1)
    rhs=om*a**-3+rg+rho_nu+floor
    drhs=-3*om*a**-3-4*rg-3*(rho_nu+p_nu)
    E=np.zeros(n);eq=np.zeros(n);waa=np.zeros(n)
    for j,A in enumerate(rhs):
        E[j]=brentq(lambda e:e*e*(1-f(e))-A,
                      np.sqrt(A)*(1-1e-11),
                      np.sqrt(A)/np.sqrt(1-lam/2)*(1+1e-11))
        der=2*E[j]*(1-f(E[j]))-E[j]*E[j]*fp(E[j])
        dedN=drhs[j]/der
        rx=floor+E[j]**2*f(E[j])
        waa[j]=-1-(2*E[j]*f(E[j])+E[j]**2*fp(E[j]))*dedN/(3*rx)
        eq[j]=(E[j]**2*(1-f(E[j]))-A)/E[j]**2
    if min(waa)<-1-1e-8:
        raise ValueError('not a canonical branch')
    return a,waa,E,max(abs(eq)),float(rho_nu[-1])

def main():
    a,w,E,error,rn=history()
    p=camb.CAMBparams()
    p.set_cosmology(H0=69.28,ombh2=.02237,omch2=.119,
                    mnu=.06,nnu=3.044,num_massive_neutrinos=1,tau=.054)
    p.InitPower.set_params(As=2.1e-9,ns=.965)
    p.set_dark_energy_w_a(a,w,dark_energy_model='fluid')
    p.set_for_lmax(4000,lens_potential_accuracy=4,
                   lens_output_margin=1250,nonlinear=False)
    r=camb.get_results(p)
    cl=r.get_cmb_power_spectra(p,lmax=4000,CMB_unit='muK',raw_cl=True)['total']
    pp=r.get_lens_potential_cls(lmax=4000,raw_cl=True)[:,0]
    ell=np.arange(len(pp))
    output=Path('results/qict_massive_nu006_camblmax4000')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'CMB_QICT_massive_nu006_C_ell.npz',
             ell=ell,tt=cl[:,0],ee=cl[:,1],bb=cl[:,2],te=cl[:,3],
             pp=pp,kk=.25*(ell*(ell+1))**2*pp)
    np.savetxt(output/'receiver_massive_w_a.csv',
               np.column_stack((a,w,E)),delimiter=',',header='a,w,E',comments='')
    payload={'status':'massive-neutrino full Boltzmann spectrum computed',
             'observed_Planck_loglike':None,'observed_ACT_loglike':None,
             'camb_version':camb.__version__,'lmax':4000,'mnu_eV':.06,
             'N_eff':3.044,'Friedmann_rel_residual_max':error,
             'Omega_nu_massive_at_a1':rn,'derived_parameters':r.get_derived_params()}
    (output/'manifest.json').write_text(json.dumps(payload,indent=2,default=float))
    print(json.dumps(payload,indent=2,default=float))
if __name__=='__main__':
    main()
