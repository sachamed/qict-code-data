"""QICT fixed-point CAMB 2.x verification, without Planck/ACT data likelihoods.

The uncoupled canonical receiver uses a CAMB DarkEnergyFluid with cs2=1.
Massless neutrinos exactly match the frozen one-pole analytic reference.
"""
import argparse, json, numpy as np
from pathlib import Path
from scipy.optimize import brentq

def history(H0=69.28,ombh2=.02237,omch2=.119,neff=3.044,
            lam=.03776, scale=1e5,n=2200):
    h=H0/100;om=(ombh2+omch2)/h**2
    rad=(2.469e-5/h**2)*(1+.22710731766*neff)
    a=np.geomspace(1e-8,1,n)
    def f(E):
        x=E/scale;return lam*x/(1+x*x)
    def fp(E):
        x=E/scale;return lam/scale*(1-x*x)/(1+x*x)**2
    floor=1-om-rad-f(1.)
    E=np.empty(n);w=np.empty(n);resid=np.empty(n)
    for i,ai in enumerate(a):
        A=om/ai**3+rad/ai**4+floor
        E[i]=brentq(lambda x:x*x*(1-f(x))-A,
                    np.sqrt(A)*(1-1e-10),np.sqrt(A)/np.sqrt(1-lam/2)*(1+1e-10))
        Fp=2*E[i]*(1-f(E[i]))-E[i]*E[i]*fp(E[i])
        dEdN=(-3*om/ai**3-4*rad/ai**4)/Fp
        rho=floor+E[i]**2*f(E[i])
        w[i]=-1-(2*E[i]*f(E[i])+E[i]**2*fp(E[i]))*dEdN/(3*rho)
        resid[i]=(E[i]**2*(1-f(E[i]))-A)/E[i]**2
    if np.min(w)<-1-1e-8:raise ValueError("Canonical branch violation")
    return a,w,E,float(np.max(abs(resid)))

def run(out='results/qict_fixed_camblmax4000',lmax=4000):
    import camb
    a,w,E,error=history()
    p=camb.CAMBparams()
    p.set_cosmology(H0=69.28,ombh2=.02237,omch2=.119,
                    mnu=0,nnu=3.044,num_massive_neutrinos=0,tau=.054)
    p.InitPower.set_params(As=2.1e-9,ns=.965)
    p.set_dark_energy_w_a(a,w,dark_energy_model='fluid')
    p.set_for_lmax(lmax,lens_potential_accuracy=4,lens_output_margin=1250,
                   nonlinear=False)
    results=camb.get_results(p)
    cls=results.get_cmb_power_spectra(p,lmax=lmax,CMB_unit='muK',raw_cl=True)['total']
    pp=results.get_lens_potential_cls(lmax=lmax,raw_cl=True)[:,0]
    ell=np.arange(len(pp))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'CMB_QICT_C_ell.npz',ell=ell,tt=cls[:,0],
        ee=cls[:,1],bb=cls[:,2],te=cls[:,3],pp=pp,
        kk=.25*(ell*(ell+1))**2*pp)
    np.savetxt(out/'receiver_w_a.csv',np.column_stack((a,w,E)),delimiter=',',
               header='a,w,E',comments='')
    report={'status':'CAMB fixed-parameter theory spectra computed',
      'observed_Planck_loglike':None,'observed_ACT_loglike':None,
      'camb_version':camb.__version__,'lmax':lmax,
      'parameters':{'H0':69.28,'ombh2':.02237,'omch2':.119,'neff':3.044,
      'mnu':0,'tau':.054,'As':2.1e-9,'ns':.965,'lam':.03776,'scale':1e5},
      'Friedmann_rel_residual_max':error,
      'derived_parameters':results.get_derived_params()}
    (out/'manifest.json').write_text(json.dumps(report,indent=2,default=float))
    print(json.dumps(report,indent=2,default=float))
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/qict_fixed_camblmax4000')
    ap.add_argument('--lmax',type=int,default=4000)
    args=ap.parse_args()
    run(args.out,args.lmax)
