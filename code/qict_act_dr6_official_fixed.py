"""Evaluate OFFICIAL ACT DR6 lensing v1.2 standalone and joint lensing data.

Uses released covariance matrices through the upstream ACT lenslike package.
Lensing-only=True and corrections=False: the released primary-CMB-marginalized
covariance is used, without adding any Planck/ACT primary likelihood.
ACT-only and ACT+Planck are ALTERNATIVE variants, not multiplied.
"""
from pathlib import Path
import json
import numpy as np
import act_dr6_lenslike as act

BASE=Path('results/qict_act_dr6_fixed_official')
BASE.mkdir(parents=True,exist_ok=True)
runs={
  'massless':'results/qict_fixed_camblmax4000/CMB_QICT_C_ell.npz',
  'massive_0p06':'results/qict_massive_nu006_camblmax4000/CMB_QICT_massive_nu006_C_ell.npz'
}
out={'data_release':'ACT DR6 lensing v1.2, NASA LAMBDA',
     'likelihood_implementation':'act_dr6_lenslike official release',
     'statistical_setting':'lens_only=True; like_corrections=False; released CMB-marginalized covariance',
     'note':'Single cosmological point per case, not posterior; variants are alternatives',
     'Planck_TTTEEE_loglike':None,'posterior_chains':None,'results':{}}
for v in ('act_baseline','actplanck_baseline'):
    data=act.load_data(v,lens_only=True,like_corrections=False)
    vr={}
    for title,path in runs.items():
        x=np.load(path)
        ell=x['ell'].astype(int)
        clkk=x['kk']
        ln=act.generic_lnlike(data,ell,clkk,ell,x['tt'],x['ee'],x['te'],x['bb'])
        vr[title]={'lnlike':float(ln),'chi2':float(-2*ln)}
    vr['delta_chi2_massive_minus_massless']=vr['massive_0p06']['chi2']-vr['massless']['chi2']
    vr['n_binned_measurements']=int(len(data['data_binned_clkk']))
    vr['covariance_shape']=list(data['cov'].shape)
    out['results'][v]=vr
(BASE/'ACT_DR6_official_fixed_point_loglikes.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
