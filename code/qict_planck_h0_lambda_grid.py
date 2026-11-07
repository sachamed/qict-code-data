"""QICT true Planck TTTEEE + lowTT/EE two-parameter conditional grid (H0,lambda)."""
from pathlib import Path
import json,numpy as np
from cobaya.model import get_model
p=Path('results/qict_planck_h0_lambda');p.mkdir(parents=True,exist_ok=True)
likes=['planck_2018_highl_plik.TTTEEE_lite_native','planck_2018_lowl.TT','planck_2018_lowl.EE']
fixed=dict(ombh2=.02237,omch2=.119,tau=.054,logA=float(np.log(21)),ns=.965,mnu=.06,A_planck=1.)
m=get_model({'packages_path':'/tmp/cobaya_packages',
 'theory':{'qict_planck_native_fixed.QICTFixedTheory':{'lmax':4200}},
 'likelihood':{k:None for k in likes},
 'params':{**fixed,'H0':{'prior':{'min':62,'max':73}},'lam':{'prior':{'min':0,'max':.10}}}})
hgrid=np.array([65.,66.,67.,67.7,68.3,69.,69.28,69.8,70.5])
lgrid=np.array([0.,.01,.02,.03,.03776,.045,.06])
rows=[]
for H0 in hgrid:
 for lam in lgrid:
  res=m.logposterior({'H0':float(H0),'lam':float(lam)},as_dict=True)
  row={'H0':float(H0),'lam':float(lam),'lnlike':float(sum(res['loglikes'].values()))}
  rows.append(row)
  print('GRID',row,flush=True)
  (p/'grid_results_partial.json').write_text(json.dumps(rows,indent=2))
best=max(rows,key=lambda r:r['lnlike'])
LCDM=max((r for r in rows if r['lam']==0),key=lambda r:r['lnlike'])
report={'likelihood':'official Planck Plik-lite TTTEEE + Commander low TT + SimAll low EE',
'status':'completed 63-point conditional grid, NOT marginalized 7D posterior',
'fixed':fixed,'H0_grid':hgrid.tolist(),'lambda_grid':lgrid.tolist(),
'best_grid_point':best,'best_lambda_zero_grid':LCDM,
'delta_chi2_best_vs_best_LCDM_grid':float(-2*(best['lnlike']-LCDM['lnlike'])),
'rows':rows}
(p/'grid_2d_planck_h0_lambda.json').write_text(json.dumps(report,indent=2))
np.savetxt(p/'grid_2d_planck_h0_lambda.csv',
 [[r['H0'],r['lam'],r['lnlike']] for r in rows],delimiter=',',header='H0,lambda,lnlike',comments='')
print(json.dumps({key:report[key] for key in
 ['best_grid_point','best_lambda_zero_grid','delta_chi2_best_vs_best_LCDM_grid']},indent=2))
