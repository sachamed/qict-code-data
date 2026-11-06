"""QICT ACT DR6 observed primary TT TE EE: official ACT-lite.
Independent primary measurement only: do not multiply by full Plik primary
without the published PlanckActCut and overlap treatment.
"""
from pathlib import Path
import json, traceback, numpy as np
from cobaya.model import get_model
out=Path('results/qict_act_primary')
out.mkdir(parents=True,exist_ok=True)
pars=dict(H0=69.28,ombh2=.02237,omch2=.119,
          tau=.054,logA=float(np.log(21)),ns=.965,
          mnu=.06,lam=.03776,A_act=1.,P_act=1.)
data={'likelihood_name':'act_dr6_cmbonly','dataset':'ACT DR6 foreground-marginalized TT TE EE',
       'model':'QICT canonical one-pole receiver, fully evaluated CAMB',
       'fixed_parameters':pars,'status':'not calculated'}
try:
    m=get_model({'packages_path':'/tmp/cobaya_packages',
         'theory':{'qict_planck_native_fixed.QICTFixedTheory':{'lmax':9000}},
         'likelihood':{'act_dr6_cmbonly':None},
         'params':pars})
    d=m.logposterior({},as_dict=True)
    data['status']='calculated'
    data['loglikes']={k:float(v) for k,v in d['loglikes'].items()}
    data['chi2']={k:float(-2*v) for k,v in d['loglikes'].items()}
except Exception as e:
    data['error']=repr(e)
    data['traceback']=traceback.format_exc()[-7000:]
(out/'ACT_DR6_primary_TTTEEE_fixed.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
if data['status']!='calculated':raise SystemExit(1)
