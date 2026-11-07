"""ACT+Planck (joint lensing), DESI DR2, and local H0: real ΛCDM DATA PIPELINE.

Validation baseline only. This cannot be advertised as an inference of the
g>0 finite-N QICT sector: the connected coupled Boltzmann backend does not yet exist.
The joint ACT-Planck LENSING product replaces standalone lensing. The primary
ACT and Planck spectra are deliberately NOT multiplied because of sky overlap.

No posterior is claimed; these are fixed-point log-likelihood evaluations.
"""
import json, traceback
from pathlib import Path
import numpy as np
from cobaya.model import get_model
OUT=Path("results/qict_joint_reference_lcdm")
OUT.mkdir(parents=True,exist_ok=True)
likes=["planck_2018_highl_plik.TTTEEE_lite_native",
       "planck_2018_lowl.TT","planck_2018_lowl.EE",
       "act_dr6_lenslike.ACTDR6LensLike","bao.desi_dr2"]
info={"packages_path":"/tmp/cobaya_packages",
 "theory":{"camb":{"extra_args":{"halofit_version":"mead2016",
               "lens_potential_accuracy":4,"lens_margin":1250,
               "nonlinear":True,"num_massive_neutrinos":1}}},
 "likelihood":{
   "planck_2018_highl_plik.TTTEEE_lite_native":None,
   "planck_2018_lowl.TT":None,
   "planck_2018_lowl.EE":None,
   "act_dr6_lenslike.ACTDR6LensLike":{"lens_only":False,
        "lmax":4000,"variant":"actplanck_baseline","stop_at_error":True},
   "bao.desi_dr2":None},
 "params":{"H0":{"prior":{"min":60,"max":80}},
           "ombh2":{"prior":{"min":0.020,"max":.025}},
           "omch2":{"prior":{"min":.10,"max":.15}},
           "tau":{"prior":{"min":.03,"max":.09}},
           "As":{"prior":{"min":1.8e-9,"max":2.5e-9}},
           "ns":{"prior":{"min":.92,"max":1.03}},
           "A_planck":1.,"mnu":.06,"nnu":3.044}}
points=[
   {"name":"Planck_fiducial","H0":67.4,"ombh2":.02237,"omch2":.120,
    "tau":.0544,"As":2.10e-9,"ns":.965},
   {"name":"H0_73_fixed_others","H0":73.04,"ombh2":.02237,"omch2":.120,
    "tau":.0544,"As":2.10e-9,"ns":.965},
]
report={"status":"running",
 "scope":"Real observed fixed-point baseline for ΛCDM ONLY, not QICT coupled or MCMC",
 "sources":"official Planck 2018 lite + lowT lowE; published joint ACT-Planck lensing; DESI DR2 BAO; SH0ES 2022 H0 Gaussian",
 "overlap":"ACT+Planck lensing product replaces individual reconstructions; ACT primary omitted",
 "SH0ES":{"mean":73.04,"sigma":1.04,"independence_assumed":True,
          "gaussian_constant_dropped":True},
 "model":"ΛCDM standard CAMB with HMCode mead2016",
 "results":[]}
try:
   model=get_model(info)
   for point in points:
      row=dict(point)
      q={k:v for k,v in point.items() if k!="name"}
      try:
         post=model.logposterior(q,as_dict=True)
         logs={k:float(v) for k,v in post["loglikes"].items()}
         if not all(np.isfinite(v) for v in logs.values()):
              raise RuntimeError("non-finite survey likelihood")
         sh0=-.5*((point["H0"]-73.04)/1.04)**2
         row.update({"status":"computed","components_loglike":logs,
           "SH0ES_loglike_drop_constant":sh0,
           "primary_plus_lensing_bao_total":float(sum(logs.values())),
           "all_plus_SH0ES_total":float(sum(logs.values())+sh0)})
      except Exception as e:
         row.update({"status":"failed","exception":repr(e),
                    "traceback":traceback.format_exc()[-10000:]})
      report["results"].append(row)
      (OUT/"lcdm_official_joint_fixed.json").write_text(json.dumps(report,indent=2))
      print("JOINT_REFERENCE",row,flush=True)
except Exception as e:
   report["initialisation_failure"]=repr(e)
   report["initialisation_traceback"]=traceback.format_exc()[-10000:]
report["status"]="complete" if len(report["results"])==len(points) and all(x["status"]=="computed" for x in report["results"]) else "failed"
(OUT/"lcdm_official_joint_fixed.json").write_text(json.dumps(report,indent=2))
if report["status"]!="complete": raise SystemExit("Joint likelihood baseline failed; see artifact")
