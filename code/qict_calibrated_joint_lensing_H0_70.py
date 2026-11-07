"""Observed fixed-input Planck 2018+ACT-Planck DR6 lensing+DESI DR2 gate.

QICT point at H0=70 is calibrated only on Planck 2018 and DESI,
so ACT-Planck joint lensing is an additional discriminating probe.
Each reference is computed with the same CLASS fork, data and neutrino
conventions. Full non-linear lensing and posterior are not implemented.
"""
import json,traceback
from pathlib import Path
import numpy as np
from cobaya.model import get_model
from qict_class_cobaya_joint import ACTDR6QICTJointLensing

OUT=Path("results/qict_calibrated_joint_lensing")
OUT.mkdir(parents=True,exist_ok=True)
OR=2.469e-5*(1+.22710731766*3.044)
OPHI=.00038644699225749197*.674**2
POINTS=[
 {"name":"LCDM_ref_67p4","kind":"lcdm","H0":67.4,"omc":.12,"ns":.965},
 {"name":"LCDM_sameparams_70","kind":"lcdm","H0":70.,"omc":.122,"ns":.975},
 {"name":"QICT_z2030_H0_70","kind":"qict","H0":70.,"omc":.122,"ns":.975,
  "V0":11.00848689636948,"F":.14332514879730884}
]
LIKES={
 "planck_2018_highl_plik.TTTEEE_lite_native":None,
 "planck_2018_lowl.TT":None,"planck_2018_lowl.EE":None,
 "qict_class_cobaya_joint.ACTDR6QICTJointLensing":None,
 "bao.desi_dr2":None
}
def run_one(p):
 h=p["H0"]/100
 extra={"Omega_scf":0,"Omega_fld":0,"gauge":"newtonian",
        "non_linear":None,"N_ncdm":0,"N_ur":3.044,"lensing":"yes"}
 params={"H0":p["H0"],"omega_b":.02237,"omega_cdm":p["omc"],
         "n_s":p["ns"],"A_s":2.1e-9,"tau_reio":.0544,"A_planck":1.}
 if p["kind"]=="qict":
  F=p["F"];V=p["V0"];params["cdm_c"]=.02
  pars=f"{V:.15g}, {F:.15g}, 250., 0., 0., 0., 0., 0., 0., 0., {2.5*F:.15g}, 0."
  extra.update({"model_cdm":"i","scf_potential":"QICT_finite_N",
    "scf_parameters":pars,"attractor_ic_scf":"no","scf_tuning_index":0,
    "Omega_scf":-1,
    "Omega_Lambda":1-(.02237+p["omc"]+OR+OPHI)/h**2})
 config={"packages_path":"/tmp/cobaya_packages","debug":False,
    "theory":{"classy":{"stop_at_error":False,"extra_args":extra}},
    "likelihood":LIKES,"params":params}
 row={"point":p}
 try:
  model=get_model(config)
  post=model.logposterior({},as_dict=True)
  components={str(k):float(v) for k,v in post["loglikes"].items()}
  if len(components)!=5 or not all(np.isfinite(v) for v in components.values()):
   raise ValueError("missing/nonfinite official data component")
  unnorm_shoes=-.5*((p["H0"]-73.04)/1.04)**2
  row.update({"status":"computed","loglikes":components,
    "total_no_SH0ES":float(sum(components.values())),
    "heldout_SH0ES_loglike":float(unnorm_shoes),
    "total_with_SH0ES_diagnostic":float(sum(components.values())+unnorm_shoes)})
 except Exception as e:
  row.update({"status":"failed","error":repr(e),
    "traceback":traceback.format_exc()[-9000:]})
 print("QICT_CALIBRATED_OBSERVED_JOINT",json.dumps(row),flush=True)
 return row

def main():
 report={
  "status":"running",
  "observations":"official Planck 2018 Plik-lite high-ell, lowTT, lowEE, joint ACT-Planck DR6 v1.2 lensing, DESI DR2 BAO",
  "ACT_lensing":"ACT-Planck actplanck_baseline lens_only=False like_corrections=True",
  "same_input_LCDM_control":True,
  "profile_likelihood_converged":False,"MCMC_converged":False,
  "nonlinear":"disabled due to CLASS SCF+HMcode incompatibility",
  "H0_70_QICT_calibrated_on":"Planck+DESI fixed-input acoustic shooting; not ACT",
  "SH0ES":"held-out Gaussian, not used to calibrate any point",
  "rows":[]
 }
 for p in POINTS:
  report["rows"].append(run_one(p))
  (OUT/"joint_lens_observed.json").write_text(json.dumps(report,indent=2)+"\n")
 good=[p for p in report["rows"] if p["status"]=="computed"]
 if len(good)==3:
  b=good[0]["total_no_SH0ES"]
  report["delta_chi2_vs_LCDM_ref_fixed"]={p["point"]["name"]:-2*(p["total_no_SH0ES"]-b) for p in good[1:]}
  report["status"]="complete"
 else: report["status"]="partial"
 (OUT/"joint_lens_observed.json").write_text(json.dumps(report,indent=2)+"\n")
 if report["status"]!="complete":raise SystemExit("Joint lensing fixed-point gate incomplete")

if __name__=="__main__":main()
