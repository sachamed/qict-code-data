"""ACT DR6 primary TT/TE/EE independent check for the exact n=3 scalar.

The Planck primary and ACT primary measured bandpowers can overlap: this
file does not multiply their likelihoods or assert a global posterior.
"""
import json,traceback
from pathlib import Path
import numpy as np
from cobaya.model import get_model
out=Path('results/qict_direct_rotor_act_primary');out.mkdir(parents=True,exist_ok=True)
cases=[dict(label='LCDM_control',H0=67.7,omch2=.12,ns=.965),
       dict(label='Rotor_best_24_grid',H0=70.5,omch2=.128,ns=.99),
       dict(label='Rotor_high_H0',H0=73.,omch2=.128,ns=.99)]
rows=[]
fixed=dict(ombh2=.02237,tau=.054,logA=float(np.log(21)),
           mnu=.06,A_act=1.,P_act=1.)
for case in cases:
    label=case['label'];params={**fixed,**{k:v for k,v in case.items() if k!='label'}}
    if label=='LCDM_control':
        info={'packages_path':'/tmp/cobaya_packages',
              'theory':{'qict_planck_native_fixed.QICTFixedTheory':{'lmax':9000}},
              'likelihood':{'act_dr6_cmbonly':None},
              'params':{**params,'lam':0.}}
    else:
        info={'packages_path':'/tmp/cobaya_packages',
              'theory':{'qict_direct_axion_planck.QICTDirectRotor':{'lmax':9000}},
              'likelihood':{'act_dr6_cmbonly':None},
              'params':{**params,'zc':3400.,'fde_zc':.1,'theta_i':2.5}}
    item={'label':label,'parameters':info['params']}
    try:
        m=get_model(info);p=m.logposterior({},as_dict=True)
        vals={k:float(v) for k,v in p['loglikes'].items()}
        if not np.all(np.isfinite(list(vals.values()))):raise RuntimeError('Invalid ACT likelihood')
        item.update({'status':'computed','loglikes':vals,
                     'lnlike':float(sum(vals.values()))})
    except Exception as e:
        item.update({'status':'failed','error':repr(e),'traceback':traceback.format_exc()[-2500:]})
    rows.append(item)
    (out/'ACT_DR6_direct_scalar_conditional.json').write_text(json.dumps(rows,indent=2))
    print('QICT_ACT_PRIMARY',label,item['status'],item.get('lnlike'),item.get('error',''),flush=True)
if all(x['status']=='computed' for x in rows):
    base=rows[0]['lnlike']
    for x in rows:x['delta_chi2_vs_LCDM_control']=float(-2*(x['lnlike']-base))
    (out/'ACT_DR6_direct_scalar_conditional.json').write_text(json.dumps(rows,indent=2))
else: raise SystemExit('Some ACT primary points failed, see artifact')
