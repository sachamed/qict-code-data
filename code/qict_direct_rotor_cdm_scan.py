"""Conditional three-coordinate Planck test: H0, physical CDM density and tilt.

The full scalar perturbations are evolved with CAMB EarlyQuintessence n=3.
This scan probes the standard EDE acoustic/peak-height degeneracy rather than
judging a nonzero receiver with all vanilla parameters artificially frozen.
The only correct claim is a conditional grid result, not a converged posterior.
"""
import json,traceback
from pathlib import Path
import numpy as np
from cobaya.model import get_model
OUT=Path('results/qict_direct_rotor_cdm_scan');OUT.mkdir(parents=True,exist_ok=True)
names=['planck_2018_highl_plik.TTTEEE_lite_native','planck_2018_lowl.TT','planck_2018_lowl.EE']
fixed={'ombh2':.02237,'tau':.054,'logA':float(np.log(21.)),
       'mnu':.06,'A_planck':1.,'theta_i':2.5,'zc':3400.,'fde_zc':.10}
info={'packages_path':'/tmp/cobaya_packages',
 'theory':{'qict_direct_axion_planck.QICTDirectRotor':{'lmax':3500}},
 'likelihood':{name:None for name in names},
 'params':{**fixed,
           'H0':{'prior':{'min':65,'max':76}},
           'omch2':{'prior':{'min':.115,'max':.16}},
           'ns':{'prior':{'min':.94,'max':1.05}}}}
model=get_model(info)
cases=[(h,oc,ns) for h in (70.5,72.0,73.0)
       for oc in (.12,.128,.136,.144)
       for ns in (.965,.99)]
rows=[]
report={'data':'Planck 2018 native Plik-lite TTTEEE + Commander low TT + SimAll low EE',
 'model':'direct CAMB scalar EarlyQuintessence n=3 QICT-compatible potential',
 'fixed_parameters':fixed,'searched_coordinates':['H0','omch2','ns'],
 'remaining_parameters_not_optimized':['ombh2','tau','As','zc','fde_zc','theta_i','A_planck'],
 'statistics':'conditional 24-point grid, not a posterior or a Bayesian evidence',
 'results':rows}
for H0,omch2,ns in cases:
    p={'H0':H0,'omch2':omch2,'ns':ns}
    try:
        rr=model.logposterior(p,as_dict=True)
        ll={k:float(v) for k,v in rr['loglikes'].items()}
        if not np.all(np.isfinite(list(ll.values()))):raise ValueError('Nonfinite measured likelihood')
        p.update({'status':'computed','lnlike':float(sum(ll.values())),'components':ll})
    except Exception as e:
        p.update({'status':'failed','error':repr(e),'traceback':traceback.format_exc()[-2000:]})
    rows.append(p)
    (OUT/'grid.json').write_text(json.dumps(report,indent=2))
    print('QICT_DEGENERACY',p,flush=True)
good=[r for r in rows if r['status']=='computed']
if good:report['best_grid_point']=max(good,key=lambda r:r['lnlike'])
report['run_status']='complete' if len(good)==len(cases) else 'partial'
(OUT/'grid.json').write_text(json.dumps(report,indent=2))
if report['run_status']!='complete':raise SystemExit('Incomplete grid; inspect diagnostics')
