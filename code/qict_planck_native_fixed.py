"""Official Planck 2018 spectra likelihoods at QICT fixed parameters.
This is a real-data evaluation, NOT a marginalized posterior, and uses the
released nuisance-marginalized Plik-lite high-l plus native Commander and
SimAll and optionally the separate native Planck lensing likelihood.
"""
import json, argparse, sys, os, traceback
from pathlib import Path
import numpy as np
from cobaya.theory import Theory
from cobaya.model import get_model
from qict_cmb_massive_boltzmann import history

class QICTFixedTheory(Theory):
    params = dict(H0=None, ombh2=None, omch2=None, tau=None, logA=None,
                  ns=None, lam=None, mnu=None)
    lmax = 4200

    def initialize(self):
        self._lmax = int(self.lmax)

    def must_provide(self, **requirements):
        for field in ['Cl','unlensed_Cl']:
            if field in requirements:
                if field=='unlensed_Cl':
                    raise ValueError('unlensed_Cl is not implemented')
                req=requirements[field]
                if isinstance(req,dict) and req:
                    self._lmax=max(self._lmax,max(int(x) for x in req.values()))
        return {}

    def calculate(self,state,want_derived=True,**pars):
        import camb
        a,w,E,err,rn=history(H0=float(pars['H0']),
                  ombh2=float(pars['ombh2']),omch2=float(pars['omch2']),
                  mnu=float(pars['mnu']),lam=float(pars['lam']))
        cp=camb.CAMBparams()
        cp.set_cosmology(H0=float(pars['H0']),ombh2=float(pars['ombh2']),
              omch2=float(pars['omch2']),tau=float(pars['tau']),
              mnu=float(pars['mnu']),nnu=3.044,
              num_massive_neutrinos=(1 if float(pars['mnu'])>0 else 0))
        cp.InitPower.set_params(As=float(np.exp(pars['logA'])*1e-10),
                                  ns=float(pars['ns']))
        if float(pars['lam'])==0:
            cp.set_dark_energy(w=-1,cs2=1,dark_energy_model='fluid')
        else:
            cp.set_dark_energy_w_a(a,w,dark_energy_model='fluid')
        cp.set_for_lmax(self._lmax,lens_potential_accuracy=4,
                     lens_output_margin=1250,nonlinear=True)
        cp.NonLinearModel.set_params(halofit_version='mead2016')
        cp.NonLinear=camb.model.NonLinear_both
        r=camb.get_results(cp)
        C=r.get_cmb_power_spectra(cp,CMB_unit='muK',raw_cl=True,
                                  lmax=self._lmax)['total']
        PP=r.get_lens_potential_cls(lmax=self._lmax,raw_cl=True)[:,0]
        ell=np.arange(len(PP))
        state['Cl']={'ell':ell,'tt':C[:,0],'ee':C[:,1],
                     'bb':C[:,2],'te':C[:,3],'pp':PP}
        state['derived']={}
        state['qict_background']={'Friedmann_relative_error':float(err)}
        return True

    def get_Cl(self,ell_factor=False,units='FIRASmuK2'):
        if units not in ('FIRASmuK2','muK2'):
            raise ValueError('Unsupported units '+str(units))
        x=self.current_state['Cl']
        if not ell_factor: return x
        ell=x['ell'].astype(float)
        f=ell*(ell+1)/(2*np.pi)
        out={'ell':x['ell']}
        for key in ('tt','ee','bb','te'):out[key]=x[key]*f
        out['pp']=x['pp']*(ell*(ell+1))**2/(2*np.pi)
        return out

def run(packages):
    cases={
        'planck_2018_highl_plik.TTTEEE_lite_native': ['planck_2018_highl_plik.TTTEEE_lite_native'],
        'planck_2018_primary': ['planck_2018_highl_plik.TTTEEE_lite_native',
               'planck_2018_lowl.TT','planck_2018_lowl.EE'],
        'planck_2018_primary_plus_planck_lensing':
              ['planck_2018_highl_plik.TTTEEE_lite_native',
               'planck_2018_lowl.TT','planck_2018_lowl.EE',
               'planck_2018_lensing.native']
    }
    const=dict(H0=69.28,ombh2=.02237,omch2=.119,tau=.054,
          logA=float(np.log(2.1e-9*1e10)),ns=.965,
          lam=.03776,mnu=.06,A_planck=1.0)
    out={'engine':'Cobaya native Planck 2018 likelihoods + CAMB',
         'highl_statistical_object':'official released foreground-marginalized Plik-lite, not multifrequency nuisance-unmarginalized Plik',
         'nonlinear_theory':'CAMB HMCode mead2016, NonLinear_both',
         'theory':'QICT one-pole decoupled canonical scalar fluid',
         'constraints':'fixed point, no MCMC posterior',
         'cmb_primary_overlap':'no ACT DR6 primary spectra added',
         'packages_path':packages,'cases':{}}
    save=Path('results/qict_planck_native')
    save.mkdir(parents=True,exist_ok=True)
    for title,names in cases.items():
        try:
            info={'packages_path':packages,
                'debug':False,
                'theory':{'qict_planck_native_fixed.QICTFixedTheory':{'lmax':4200}},
                'likelihood':{name:None for name in names},
                'params':const}
            model=get_model(info)
            post=model.logposterior({},as_dict=True)
            row={'status':'calculated',
                'loglikes':{name:float(v) for name,v in post['loglikes'].items()},
                'total_loglike':float(sum(post['loglikes'].values()))}
            out['cases'][title]=row
        except Exception as e:
            out['cases'][title]={'status':'failed',
                 'error':repr(e),'traceback':traceback.format_exc()[-5000:]}
        (save/'planck_official_native_fixed_points.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    if not all(v['status']=='calculated' for v in out['cases'].values()):
        raise SystemExit('Some official Planck likelihood components failed; exact diagnostics saved.')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--packages-path',default='/tmp/cobaya_packages')
    run(ap.parse_args().packages_path)
