"""QICT initial-phase scan at matched acoustic angle and actual scalar peak.

Each theta_i gets its OWN native 2D shooting for V0,F at fixed
H0, omega_cdm, ns, zpeak=3500 and 100theta_s=1.04109.
Official Planck 2018 Plik-lite+lowTT+lowEE+DESI DR2 follows.

This is a sparse conditional phase screen, not a marginalized posterior
or an independently optimized LCDM comparison.
"""
import json,math,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from qict_peak_calibrated_observed import official
import qict_peak_calibrated_observed as base

OUTPUT=Path("results/qict_phase_scan")
OUTPUT.mkdir(parents=True,exist_ok=True)
CASES=[
 dict(name="H0_70p5",H0=70.5,omega_cdm=.124,ns=.975,zc=3500.),
 dict(name="H0_72",H0=72.,omega_cdm=.13,ns=.99,zc=3500.),
]
THETAS=[2.15,2.35,2.5,2.7,2.85,3.0,3.1]
TARGET_THETA=1.04109
SEEDS={"H0_70p5":(122.778,.176747),"H0_72":(235.52,.237342)}

def main():
 report={"status":"running",
  "observed_data":"official Planck2018 Plik-lite TTTEEE, low-TT/EE, DESI DR2 BAO",
  "method":"at every phase, shoot V0,F for z_peak=3500 and 100*theta_s=1.04109; examine resulting fpeak",
  "fixed_early":"N=250,g=0.02, 0 neutrino mass, N_ur=3.044, tau_reio=0.0544",
  "theta_i_grid":THETAS,"results":[],"posterior":False,"converged_global_fit":False}
 for case in CASES:
  for theta in THETAS:
   base.THETA=float(theta)
   row={"case":case,"theta_i":theta,"trials":[]}
   start=time.monotonic()
   V0,F0=SEEDS[case["name"]]
   try:
    # For a varied phase, keep initial potential density comparable
    # to the reference phase until full nonlinear shooting.
    x=1-math.cos(theta);xr=1-math.cos(2.5)
    def p(x):return 4*x/250**2+6*249*x*x/250**2+249*248*x**3/250**2
    V0=V0*p(xr)/p(x)
   except Exception:pass
   def obj(u):
    V,F=map(float,np.exp(u))
    try:
     r=base.physical(case,V,F)
     x=float((r["100*theta_s"]-TARGET_THETA)*50)
     y=float(np.log(r["peak_z"]/case["zc"]))
     if not (.001<r["peak_f"]<.35) or abs(r["H0_rel_error"])>.001:y+=3.
     residual=[x,y]
     row["trials"].append({"V0":V,"F":F,**r,"loss":float(np.linalg.norm(residual))})
    except Exception as e:
     residual=[5.,5.]
     row["trials"].append({"V0":V,"F":F,"error":repr(e)})
    return residual
   try:
    opt=least_squares(obj,np.log([V0,F0]),bounds=(np.log([5.,.05]),np.log([5000.,.95])),
       diff_step=.04,max_nfev=32,ftol=.003,xtol=.003,gtol=.003)
    good=[r for r in row["trials"] if "loss" in r]
    if not good:raise RuntimeError("No valid physical background")
    best=min(good,key=lambda r:r["loss"])
    row["best"]={k:v for k,v in best.items() if k not in ("background_headers",)}
    row["optimizer_message"]=str(opt.message)
    if abs(best["100*theta_s"]-TARGET_THETA)>.0009 or abs(best["peak_z"]/case["zc"]-1)>.07:
     row["status"]="rejected_acoustic_or_peak"
    elif best["peak_f"]>.3 or abs(best["H0_rel_error"])>.001:
     row["status"]="rejected_fraction_or_Friedmann"
    else:
     row["likelihood"]=official(case,best["V0"],best["F"])
     row["status"]="computed"
   except Exception as e:
    row["status"]="failed";row["error"]=repr(e)
    row["traceback"]=traceback.format_exc()[-2800:]
   row["seconds"]=round(time.monotonic()-start,1)
   report["results"].append(row)
   (OUTPUT/"phase_scan.json").write_text(json.dumps(report,indent=2)+"\n")
   print("QICT_PHASE_SCAN",json.dumps({k:v for k,v in row.items() if k!="trials"}),flush=True)
 report["status"]="complete" if all(z["status"]=="computed" for z in report["results"]) else "partial"
 good=[r for r in report["results"] if r["status"]=="computed"]
 if good:
  report["best_by_loglike"]=max(good,key=lambda r:r["likelihood"]["sum"])
 (OUTPUT/"phase_scan.json").write_text(json.dumps(report,indent=2)+"\n")
 if not good:raise SystemExit("No phase produced a valid likelihood")

if __name__=="__main__":main()
