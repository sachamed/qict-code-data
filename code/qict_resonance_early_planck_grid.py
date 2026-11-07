"""QICT pre-recombination resonance scan on REAL Planck TTTEEE, low TT and low EE.

Change the epoch of the resonant receiver, not only its amplitude.
Tests whether concentrating its positive energy near equality, rather than
at last scattering, lowers Planck penalties at high H0.

Only conditional grids: other vanilla cosmological parameters held fixed.
"""
from pathlib import Path
import numpy as np,json,traceback
from cobaya.model import get_model
from qict_resonant_cmb_theory import history
outdir=Path('results/qict_resonance_early_era')
outdir.mkdir(parents=True,exist_ok=True)
likes=['planck_2018_highl_plik.TTTEEE_lite_native',
       'planck_2018_lowl.TT','planck_2018_lowl.EE']
fixed=dict(ombh2=.02237,omch2=.119,tau=.054,
           logA=float(np.log(21.)),ns=.965,mnu=.06,A_planck=1.)
centers=[2500,3000,3400]
hgrid=[67.7,70.,71.5,73.]
amps=[.10,.20,.30]
report={'status':'in_progress','likelihoods':likes,'g':1.6,
        'fixed':fixed,'z_center_grid':centers,
        'H0_grid':hgrid,'peak_amplitude_grid':amps,
        'description':'37 real Planck data evaluations, 36 QICT early-era resonance plus LambdaCDM control',
        'assumptions':'independent canonical scalar effective-fluid closure cs2=1; resonance constitutive ansatz',
        'rows':[]}
rows=report['rows']
baseline_model=get_model({'packages_path':'/tmp/cobaya_packages',
       'theory':{'qict_resonant_cmb_theory.ResonantQICTTheory':{'lmax':4200}},
       'likelihood':{name:None for name in likes},
       'params':{**fixed,'H0':{'prior':{'min':64,'max':77}},
                       'A_res':{'prior':{'min':0,'max':.45}}}})
b=baseline_model.logposterior({'H0':67.7,'A_res':0.},as_dict=True)
base={k:float(v) for k,v in b['loglikes'].items()}
if not np.all(np.isfinite(list(base.values()))):raise RuntimeError('LambdaCDM control invalid')
control=sum(base.values())
rows.append({'status':'computed','H0':67.7,'A_res':0.,'center_z':None,
             'lnlike':float(control),'components':base})
for zc in centers:
    hc=100*np.sqrt(.143*(1+zc)**3+4.18e-5*(1+zc)**4)
    model=get_model({'packages_path':'/tmp/cobaya_packages',
       'theory':{'qict_resonant_cmb_theory.ResonantQICTTheory':{
                'lmax':4200,'damping_ratio':1.6,'Hc':float(hc)}},
       'likelihood':{name:None for name in likes},
       'params':{**fixed,'H0':{'prior':{'min':64,'max':77}},
                       'A_res':{'prior':{'min':0,'max':.45}}}})
    for H0 in hgrid:
      for A in amps:
        row={'H0':H0,'A_res':A,'center_z':zc,'Hc_km_s_Mpc':float(hc)}
        try:
          a,w,E,fr,err,floor=history(H0=H0,A=A,Hc=hc)
          p=model.logposterior({'H0':H0,'A_res':A},as_dict=True)
          comp={name:float(v) for name,v in p['loglikes'].items()}
          if not np.all(np.isfinite(list(comp.values()))):raise RuntimeError('Planck likelihood not finite')
          row.update({'status':'computed','components':comp,'lnlike':float(sum(comp.values())),
              'delta_chi2_vs_LCDM_control':float(-2*(sum(comp.values())-control)),
              'wmax':float(np.max(w)),'f_1090':float(np.interp(1/1091,a,fr)),
              'f_3400':float(np.interp(1/3401,a,fr)),'Friedmann_error':err})
        except Exception as e:
          row.update({'status':'invalid','error':repr(e),
                      'traceback':traceback.format_exc()[-1600:]})
        rows.append(row)
        (outdir/'planck_early_resonance_grid.json').write_text(json.dumps(report,indent=2))
        print('POINT',zc,H0,A,row['status'],row.get('lnlike'),
              row.get('error',''),flush=True)
good=[x for x in rows if x['status']=='computed']
best=max(good,key=lambda x:x['lnlike'])
report['best_computed_point']=best
report['baseline_LCDM_lnlike']=float(control)
report['status']='complete' if len(good)==1+len(centers)*len(hgrid)*len(amps) else 'partial'
(outdir/'planck_early_resonance_grid.json').write_text(json.dumps(report,indent=2))
print('BEST',best,flush=True)
if report['status']!='complete':raise SystemExit('At least one likelihood invalid; see artifact')
