"""Observed high-energy QICT fixed-point check: Planck 2018, ACT-Planck DR6, DESI DR2.

All imported likelihoods use their official data. Nonlinear HMcode is deliberately
not used because this CLASS fork rejects SCF+HMcode. These are LINEAR-lensing
research points and are not a substitute for a complete nonlinear final fit.
SH0ES is a separate diagnostic; no sampled posterior is implied.
"""
from pathlib import Path
from copy import deepcopy
import traceback,json
import numpy as np
from cobaya.model import get_model
from qict_class_cobaya_joint import ACTDR6QICTJointLensing

H0_GRID=(67.4,74.93)
NEFF=3.044
OMEGA_M=.02237+.12
OMEGA_R=2.469e-5*(1+.22710731766*NEFF)
OMEGA_SCF_PHYSICAL=.00038644699225749197*.674**2
MODEL_PARAMETERS="201.55504807115258, 0.225, 250., 0., 0., 0., 0., 0., 0., 0., 0.5625, 0."

def info_for(h0):
    h=h0/100.
    omega_lam=1-(OMEGA_M+OMEGA_R+OMEGA_SCF_PHYSICAL)/h**2
    return {
        "packages_path":"/tmp/cobaya_packages",
        "debug":True,
        "theory":{"classy":{
            "stop_at_error":True,
            "extra_args":{
                "model_cdm":"i","scf_potential":"QICT_finite_N",
                "scf_parameters":MODEL_PARAMETERS,
                "attractor_ic_scf":"no","scf_tuning_index":0,
                "Omega_scf":-1,"Omega_Lambda":omega_lam,
                "Omega_fld":0,"gauge":"newtonian",
                "N_ncdm":0,"N_ur":NEFF,
                "lensing":"yes",
            }
        }},
        "likelihood":{
            "planck_2018_highl_plik.TTTEEE_lite_native":None,
            "planck_2018_lowl.TT":None,
            "planck_2018_lowl.EE":None,
            "qict_class_cobaya_joint.ACTDR6QICTJointLensing":None,
            "bao.desi_dr2":None
        },
        "params":{
            "H0":h0,"omega_b":.02237,"omega_cdm":.12,
            "tau_reio":.0544,"A_s":2.1e-9,"n_s":.965,
            "A_planck":1.0,
            "cdm_c":.02
        }
    },omega_lam

def run():
    directory=Path("results/qict_high_energy_joint_observed")
    directory.mkdir(parents=True,exist_ok=True)
    report={"status":"running","fixed_point_only":True,
      "posterior_converged":False,"nonlinear_lensing":"disabled: CLASS scf-HMCode unsupported",
      "H0_values":list(H0_GRID),
      "H0_74_93_source":"independent six-point native CLASS geometric acoustic scan, not a posterior",
      "dataset":"official Planck 2018 Plik-lite TTTEEE, low-TT, low-EE, joint ACT-Planck DR6 v1.2 lensing, DESI DR2 BAO",
      "SH0ES":"held-out Gaussian penalty (73.04 +/- 1.04), no normalization",
      "results":[]}
    for h0 in H0_GRID:
        config,lam=info_for(h0)
        row={"H0":h0,"Omega_Lambda_input":lam}
        try:
            model=get_model(config)
            val=model.logposterior(as_dict=True)
            likes={str(k):float(v) for k,v in val["loglikes"].items()}
            if not likes or not all(np.isfinite(v) for v in likes.values()):
                raise ValueError("At least one official likelihood is non-finite")
            penalty=-.5*((h0-73.04)/1.04)**2
            row.update({"status":"computed","components_loglikes":likes,
                "sum_CMB_lensing_BAO_lnlike":float(sum(likes.values())),
                "SH0ES_heldout_loglike":float(penalty),
                "sum_with_local_H0_diagnostic":float(sum(likes.values())+penalty)})
        except Exception as exc:
            row.update({"status":"failed","exception":repr(exc),
              "traceback":traceback.format_exc()[-12000:]})
        report["results"].append(row)
        print("QICT_HIGH_ENERGY_OFFICIAL_LIKELIHOOD",json.dumps(row),flush=True)
        (directory/"QICT_HIGH_ENERGY_LIKELIHOODS.json").write_text(json.dumps(report,indent=2)+"\n")
    report["status"]="complete" if len(report["results"])==len(H0_GRID) and all(row["status"]=="computed" for row in report["results"]) else "failed"
    (directory/"QICT_HIGH_ENERGY_LIKELIHOODS.json").write_text(json.dumps(report,indent=2)+"\n")
    if report["status"]!="complete":raise SystemExit("Official likelihood gate incomplete; inspect report")
if __name__=="__main__":run()
