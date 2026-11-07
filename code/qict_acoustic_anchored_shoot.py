"""Acoustic-anchored QICT finite-N shooting: match theta* and peak redshift.

Two physical constraints determine F and V0 at fixed {H0,omega_c,ns}.
The energy fraction f_peak is an OUTPUT. Official Planck+DESI likelihoods
are a second gate. This is not a sampled posterior or global profile.
"""
import json,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from qict_peak_calibrated_observed import physical,official

OUT=Path("results/qict_acoustic_shoot")
OUT.mkdir(parents=True,exist_ok=True)
THETA_STAR=1.04109
CASES=[
 dict(name="H0_70p5",H0=70.5,omega_cdm=.124,ns=.975,zc=3500.,fpeak=None),
 dict(name="H0_72",H0=72.,omega_cdm=.13,ns=.99,zc=3500.,fpeak=None),
 dict(name="H0_73",H0=73.04,omega_cdm=.138,ns=1.,zc=3500.,fpeak=None)
]
def main():
 report={"status":"running","constraint":"actual CLASS 100*theta_s=1.04109 and scalar fraction peak at redshift 3500",
  "free_micro":"F,V0","derived_micro":"f_peak","g":.02,"N":250,
  "nonlinear_lensing":"off","likelihood":"official Planck2018+DESI DR2",
  "not_a_posterior_or_model_selection":True,"rows":[]}
 baseline=dict(H0=67.4,omega_cdm=.12,ns=.965,zc=3500,fpeak=None)
 try:report["LCDM_fixed"]=official(baseline,0.,.225,standard=True)
 except Exception as e:report["LCDM_fixed"]={"error":repr(e)}
 for target in CASES:
  row={"target":target,"trials":[]}
  t0=time.monotonic()
  def residual(logx):
   V,F=np.exp(logx)
   try:
    d=physical(target,V,F)
    r=[(d["100*theta_s"]-THETA_STAR)*50,
       np.log(d["peak_z"]/target["zc"])]
    if not .001<d["peak_f"]<.3:r[1]+=2
    if abs(d["H0_rel_error"])>.001:r[1]+=2
    row["trials"].append({"V0":float(V),"F":float(F),**d,"norm":float(np.linalg.norm(r))})
   except Exception as exc:
    r=[5,5]
    row["trials"].append({"V0":float(V),"F":float(F),"error":repr(exc)})
   print("QICT_ACOUSTIC_TRIAL",target["name"],float(V),float(F),r,flush=True)
   return np.asarray(r)
  try:
   fit=least_squares(residual,np.log([195.,.225]),
                     bounds=(np.log([10.,.08]),np.log([3000.,.8])),
                     diff_step=.045,max_nfev=32,
                     ftol=.003,xtol=.003,gtol=.003)
   good=[x for x in row["trials"] if "norm" in x]
   if not good:raise RuntimeError("No valid CLASS backgrounds")
   best=min(good,key=lambda x:x["norm"])
   row["best"]=best;row["optimizer_message"]=str(fit.message)
   row["theta_star_target_residual"]=best["100*theta_s"]-THETA_STAR
   row["peak_z_relative_residual"]=best["peak_z"]/target["zc"]-1
   row["f_peak_predicted"]=best["peak_f"]
   if abs(best["H0_rel_error"])>.001:
    row["status"]="failed_friedmann_closure"
   elif abs(row["theta_star_target_residual"])>.0008 or abs(row["peak_z_relative_residual"])>.1:
    row["status"]="no_joint_acoustic_peak_solution_at_current_parameters"
   elif not (.001<best["peak_f"]<.25):
    row["status"]="unphysical_peak_fraction_for_this_screen"
   else:
    row["likelihood"]=official(target,best["V0"],best["F"])
    row["status"]="computed"
    if "sum" in report.get("LCDM_fixed",{}):
     row["delta_chi2_vs_LCDM_fixed"]=-2*(row["likelihood"]["sum"]-report["LCDM_fixed"]["sum"])
  except Exception as e:
   row["status"]="failed";row["error"]=repr(e);row["traceback"]=traceback.format_exc()[-4000:]
  row["wall_seconds"]=round(time.monotonic()-t0,1)
  report["rows"].append(row)
  (OUT/"acoustic_fit.json").write_text(json.dumps(report,indent=2)+"\n")
  print("QICT_ACOUSTIC_FINAL",json.dumps({k:v for k,v in row.items() if k!="trials"}),flush=True)
 report["status"]="complete" if all(x["status"]=="computed" for x in report["rows"]) else "partial"
 (OUT/"acoustic_fit.json").write_text(json.dumps(report,indent=2)+"\n")
 if not any(x["status"]=="computed" for x in report["rows"]):
  raise SystemExit("No physically valid dual acoustic/peak model; see artifact")

if __name__=="__main__":main()
