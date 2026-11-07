"""Official Planck 2018 and DESI DR2 high-energy QICT fixed-point diagnostic.
The full ACT-Planck lensing likelihood is evaluated separately in the joint workflow.
This run is not an MCMC fit or a full nonlinear-theory evaluation.
"""
from pathlib import Path
import numpy as np,traceback,json
from cobaya.model import get_model
H_GRID=[67.4,74.93]
p="201.55504807115258, 0.225, 250., 0., 0., 0., 0., 0., 0., 0., 0.5625, 0."
omega_m=.02237+.12
omega_r=2.469e-5*(1+.22710731766*3.044)
omega_phi=.00038644699225749197*.674**2
out=Path("results/qict_high_planck_bao");out.mkdir(parents=True,exist_ok=True)
d={"status":"running","scope":"Planck2018 Plik-lite TTTEEE+lowTT+lowEE+official DESI DR2, no ACT yet",
   "nonlinear":"none","H0_values":H_GRID,"fit_type":"fixed parameter points, not optimized posterior","results":[]}
for H0 in H_GRID:
 h=H0/100
 lam=1-(omega_m+omega_r+omega_phi)/h**2
 info={"packages_path":"/tmp/cobaya_packages",
   "debug":True,
   "theory":{"classy":{"stop_at_error":True,"extra_args":{
      "model_cdm":"i","scf_potential":"QICT_finite_N","scf_parameters":p,
      "attractor_ic_scf":"no","scf_tuning_index":0,
      "Omega_scf":-1,"Omega_Lambda":lam,"Omega_fld":0,
      "gauge":"newtonian","non_linear":None,
      "N_ncdm":0,"N_ur":3.044,"lensing":"yes"}}},
   "likelihood":{
      "planck_2018_highl_plik.TTTEEE_lite_native":None,
      "planck_2018_lowl.TT":None,"planck_2018_lowl.EE":None,
      "bao.desi_dr2":None},
   "params":{"H0":H0,"omega_b":.02237,"omega_cdm":.12,"tau_reio":.0544,
     "A_s":2.1e-9,"n_s":.965,"A_planck":1.,"cdm_c":.02}}
 row={"H0_input":H0,"Omega_Lambda":lam}
 try:
  model=get_model(info)
  val=model.logposterior({},as_dict=True)
  likes={str(k):float(v) for k,v in val["loglikes"].items()}
  if not likes or not all(np.isfinite(v) for v in likes.values()): raise ValueError("a likelihood is not finite")
  row.update({"status":"computed","loglikes":likes,"total_without_SH0ES":float(sum(likes.values())),
     "SH0ES_heldout":-.5*((H0-73.04)/1.04)**2})
 except Exception as ex:
  row.update({"status":"failed","error":repr(ex),"traceback":traceback.format_exc()[-10000:]})
 print("QICT_PLANCK_DESI_EVALUATION",json.dumps(row),flush=True)
 d["results"].append(row)
 (out/"planck_desi_fixed_point.json").write_text(json.dumps(d,indent=2)+"\n")
d["status"]="complete" if all(row["status"]=="computed" for row in d["results"]) else "failed"
(out/"planck_desi_fixed_point.json").write_text(json.dumps(d,indent=2)+"\n")
if d["status"]!="complete": raise SystemExit("Planck DESI likelihood evaluation incomplete")
