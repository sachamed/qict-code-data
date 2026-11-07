"""Fail-closed direct CAMB spectrum and Cobaya prior/theory diagnostics."""
import numpy as np, traceback, json
from pathlib import Path
from qict_resonant_cmb_theory import history,ResonantQICTTheory,HC_FID
from cobaya.model import get_model
import camb
pout=Path('results/qict_resonance_probe');pout.mkdir(parents=True,exist_ok=True)
data={'test':'A=0.1 H0=67.7 finite CAMB and Planck native',
      'cases':[]}
for amplitude in [0,.01,.1,.2]:
    row={'A':amplitude}
    try:
        a,w,E,f,resid,floor=history(A=amplitude,H0=67.7)
        p=camb.CAMBparams()
        p.set_cosmology(H0=67.7,ombh2=.02237,omch2=.119,
                        mnu=.06,nnu=3.044,num_massive_neutrinos=1,tau=.054)
        p.InitPower.set_params(As=2.1e-9,ns=.965)
        if amplitude==0:p.set_dark_energy(w=-1,cs2=1,dark_energy_model='fluid')
        else:p.set_dark_energy_w_a(a,w,dark_energy_model='fluid')
        p.set_for_lmax(4000,lens_potential_accuracy=4,lens_output_margin=1250,nonlinear=True)
        p.NonLinearModel.set_params(halofit_version='mead2016')
        p.NonLinear=camb.model.NonLinear_both
        r=camb.get_results(p)
        cl=r.get_cmb_power_spectra(p,lmax=4000,CMB_unit='muK',raw_cl=True)['total']
        row.update({'direct_camb_status':'success','isfinite':bool(np.all(np.isfinite(cl))),
                    'wmax':float(max(w)),'TT_200':float(cl[200,0]),
                    'derived':r.get_derived_params()})
    except Exception as e:
        row['direct_camb_status']='failure';row['error']=repr(e)
        row['trace']=traceback.format_exc()[-5000:]
    data['cases'].append(row)
    print('PROBE',amplitude,row['direct_camb_status'],
          row.get('error',''),flush=True)
(pout/'resonance_direct_probe.json').write_text(json.dumps(data,indent=2,default=float))
if any(x['direct_camb_status']=='failure' for x in data['cases']):
    raise SystemExit("Some direct CAMB spectra failed, see artifact")
