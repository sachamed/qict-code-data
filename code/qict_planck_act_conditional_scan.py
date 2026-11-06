"""Conditional Planck TTTEEE low-T/E and ACT DR6 lensing λ scan."""
from pathlib import Path
import json
import numpy as np
import act_dr6_lenslike as act
from cobaya.model import get_model

OUT=Path('results/qict_planck_act_conditional')
OUT.mkdir(parents=True,exist_ok=True)
primary=['planck_2018_highl_plik.TTTEEE_lite_native',
         'planck_2018_lowl.TT','planck_2018_lowl.EE']
fixed=dict(H0=69.28,ombh2=.02237,omch2=.119,tau=.054,
           logA=float(np.log(21.0)),ns=.965,mnu=.06,A_planck=1.)
info={'packages_path':'/tmp/cobaya_packages',
      'theory':{'qict_planck_native_fixed.QICTFixedTheory':{'lmax':4200}},
      'likelihood':{name:None for name in primary},
      'params':{**fixed,'lam':{'prior':{'min':0,'max':0.11},
                             'ref':0.03776,'proposal':0.01}}}
model=get_model(info)
data={v:act.load_data(v,lens_only=True,like_corrections=False)
      for v in ['act_baseline','actplanck_baseline']}
lam_grid=np.unique(np.r_[np.linspace(0,.1,11),0.03776])
rows=[]
for lam in lam_grid:
    p=model.logposterior({'lam':float(lam)},as_dict=True)
    llp=sum(p['loglikes'].values())
    cl=model.provider.get_Cl(ell_factor=False,units='muK2')
    ell=cl['ell'].astype(int)
    kk=0.25*(ell*(ell+1))**2*cl['pp']
    A={v:float(act.generic_lnlike(d,ell,kk,ell,cl['tt'],cl['ee'],cl['te'],cl['bb']))
       for v,d in data.items()}
    rows.append({'lambda':float(lam),'planck_primary_lnlike':float(llp),
                'planck_components':{k:float(v) for k,v in p['loglikes'].items()},
                'act_dr6_lens_lnlike':A['act_baseline'],
                'act_planck_joint_lens_lnlike':A['actplanck_baseline'],
                'planck_plus_act_lnlike':float(llp+A['act_baseline']),
                'planck_plus_joint_lens_lnlike':float(llp+A['actplanck_baseline'])})
    print('SCAN',lam,rows[-1]['planck_plus_joint_lens_lnlike'],flush=True)
base=next(r for r in rows if r['lambda']==0)
tags=['planck_primary_lnlike','planck_plus_act_lnlike','planck_plus_joint_lens_lnlike']
for row in rows:
    for tag in tags:
        row['delta_chi2_'+tag]=float(-2*(row[tag]-base[tag]))
report={'method':'fixed-six-parameter conditional scan, NOT full joint MCMC',
        'likelihoods':'official Planck Plik-lite, Commander, SimAll + ACT DR6 v1.2 lensing',
        'ACT_overlap':'ACT-only and released ACT-Planck joint lensing alternatives',
        'fixed_parameters':fixed,'lambda_grid':lam_grid.tolist(),
        'results':rows,
        'grid_best_lambda':{tag:max(rows,key=lambda r:r[tag])['lambda'] for tag in tags}}
(OUT/'conditional_lambda_scan.json').write_text(json.dumps(report,indent=2))
np.savetxt(OUT/'conditional_lambda_scan.csv',
           [[r['lambda'],r['planck_primary_lnlike'],r['planck_plus_act_lnlike'],
             r['planck_plus_joint_lens_lnlike']] for r in rows],
           delimiter=',',header='lambda,lnlike_planck,lnlike_planckACT,lnlike_planck_joint',comments='')
print(json.dumps(report['grid_best_lambda'],indent=2))
