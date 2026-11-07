"""Official ACT DR6 primary-CMB-only test of native calibrated QICT CLASS.

The ACT DR6 primary TT/TE/EE likelihood is used separately.
Do NOT combine these numbers with Plik-lite or ACT-Planck lensing without
validated covariance, overlap cuts, priors and ACT nuisance refit.
Models are fixed input points, not ACT-marginalized parameter inference.
"""
import json, math, traceback
from pathlib import Path
import numpy as np
from cobaya.model import get_model

OUT=Path("results/qict_act_dr6_primary_qict")
OUT.mkdir(parents=True,exist_ok=True)
OMEGA_R=2.469e-5*(1+.22710731766*3.044)
OMEGA_PHI_PHYSICAL=.00038644699225749197*.674**2
POINTS=[
 dict(name="LCDM_H0_67p4",kind="lcdm",H0=67.4,omega_cdm=.12,n_s=.965),
 dict(name="QICT_H0_68p5",kind="qict",H0=68.5,omega_cdm=.121,n_s=.967,V0=36.699909254729036,F=.1001613464888732),
 dict(name="QICT_H0_70p5",kind="qict",H0=70.5,omega_cdm=.124,n_s=.975,V0=122.77781813186203,F=.17674669285161573),
 dict(name="QICT_H0_72",kind="qict",H0=72.,omega_cdm=.13,n_s=.99,V0=235.51950122406512,F=.23734168553381624)
]
def one(p):
 h=p["H0"]/100.
 args={
  "Omega_fld":0,"gauge":"newtonian","non_linear":None,
  "N_ncdm":0,"N_ur":3.044,"lensing":"yes",
 }
 if p["kind"]=="lcdm":
  args["Omega_scf"]=0
 else:
  params=f'{p["V0"]:.15g}, {p["F"]:.15g}, 250., 0., 0., 0., 0., 0., 0., 0., {2.5*p["F"]:.15g}, 0.'
  args.update({"model_cdm":"i","scf_potential":"QICT_finite_N",
   "scf_parameters":params,"attractor_ic_scf":"no","scf_tuning_index":0,
   "Omega_scf":-1,
   "Omega_Lambda":1-(.02237+p["omega_cdm"]+OMEGA_R+OMEGA_PHI_PHYSICAL)/h**2})
 pars={
  "H0":p["H0"],"omega_b":.02237,"omega_cdm":p["omega_cdm"],
  "n_s":p["n_s"],"A_s":2.1e-9,"tau_reio":.0544,
  "A_act":1.0,"P_act":1.0
 }
 if p["kind"]!="lcdm":pars["cdm_c"]=.02
 cfg={"packages_path":"/tmp/cobaya_packages","debug":False,
  "theory":{"classy":{"stop_at_error":False,"extra_args":args}},
  "likelihood":{"act_dr6_cmbonly":None},
  "params":pars}
 row={"point":p,"status":"starting"}
 try:
  model=get_model(cfg)
  ln=model.logposterior({},as_dict=True)
  likes={k:float(v) for k,v in ln["loglikes"].items()}
  if not likes or not all(np.isfinite(v) for v in likes.values()):
   raise RuntimeError("ACT primary likelihood nonfinite")
  row.update({"status":"computed","loglikes":likes,"sum":float(sum(likes.values()))})
 except Exception as e:
  row.update({"status":"failed","error":repr(e),"traceback":traceback.format_exc()[-9500:]})
 print("QICT_ACT_DR6_PRIMARY",json.dumps(row),flush=True)
 return row

def main():
 report={"status":"starting","data":"official ACT Collaboration DR6 CMB-lite primary TT/TE/EE",
  "covariance":"ACT-only, no Planck primary or lensing covariance combined",
  "model":"patched native interacting CLASS linear-lensing (HMCode unsupported with SCF)",
  "type":"four fixed-parameter points, no nuisance optimization or sampled posterior",
  "points":[]}
 for p in POINTS:
  report["points"].append(one(p))
  (OUT/"primary_act_dr6.json").write_text(json.dumps(report,indent=2)+"\n")
 good=[x for x in report["points"] if x["status"]=="computed"]
 if len(good)==4:
  b=good[0]["sum"]
  report["delta_chi2_against_fixed_LCDM"]={x["point"]["name"]:-2*(x["sum"]-b) for x in good[1:]}
  report["status"]="complete"
 else:report["status"]="partial"
 (OUT/"primary_act_dr6.json").write_text(json.dumps(report,indent=2)+"\n")
 if report["status"]!="complete":raise SystemExit("ACT DR6 primary fixed-point test incomplete")

if __name__=="__main__":main()
