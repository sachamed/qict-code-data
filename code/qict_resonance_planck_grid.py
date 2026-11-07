"""Real Planck 2018 likelihood screen for the causal QICT receiver resonance.

A strict 4x4 conditional grid in H0 and resonance peak A, fixed matter/primordial
parameters; no posteriors and no claims of model selection. Acts as a fast
falsification test before a high-dimensional official fit.
"""
from pathlib import Path
import json,traceback
import numpy as np
from cobaya.model import get_model
from qict_resonant_cmb_theory import history,HC_FID
OUT=Path("results/qict_resonance_planck")
OUT.mkdir(parents=True,exist_ok=True)
likes=["planck_2018_highl_plik.TTTEEE_lite_native",
       "planck_2018_lowl.TT","planck_2018_lowl.EE"]
fixed={"ombh2":.02237,"omch2":.119,"tau":.054,
       "logA":float(np.log(21)),"ns":.965,"mnu":.06,"A_planck":1.}
info={"packages_path":"/tmp/cobaya_packages",
      "theory":{"qict_resonant_cmb_theory.ResonantQICTTheory":{
              "lmax":4200,"damping_ratio":1.6,"Hc":HC_FID}},
      "likelihood":{name:None for name in likes},
      "params":{**fixed,"H0":{"prior":{"min":64,"max":76}},
                "A_res":{"prior":{"min":0,"max":.45}}}}
model=get_model(info)
hgrid=[67.7,69.28,71.,73.]
agrid=[0.,.10,.20,.30]
rows=[]
out={"status":"in_progress","description":"Conditional Planck Plik-lite TTTEEE, low TT and low EE likelihood",
     "Hc_km_s_Mpc":HC_FID,"gamma_over_Hc":1.6,
     "fixed_parameters":fixed,"H0_grid":hgrid,"A_res_grid":agrid,
     "perturbation_closure":"canonical scalar effective fluid: cs_rest2=1, shear=0",
     "warnings":["The resonance-to-energy-density relation is a constitutive model assumption.",
        "w>1 entails negative reconstructed scalar potential on that interval.",
        "Only two coordinates scanned; does not establish full Bayesian evidence."],
     "rows":rows}
for H0 in hgrid:
  for A in agrid:
    row={"H0":H0,"A_res":A}
    try:
        a,w,E,f,err,floor=history(H0=H0,A=A)
        p=model.logposterior({"H0":H0,"A_res":A},as_dict=True)
        ll={name:float(v) for name,v in p["loglikes"].items()}
        if not np.all(np.isfinite(list(ll.values()))):
            raise RuntimeError("Planck likelihood not finite: numerical evaluation failed; no statistical comparison")
        row.update({"status":"computed","loglikes":ll,
          "planck_primary_lnlike":float(sum(ll.values())),
          "max_w":float(np.max(w)),"min_w":float(np.min(w)),
          "f_1090":float(np.interp(1/1091,a,f)),
          "f_3400":float(np.interp(1/3401,a,f)),
          "max_Friedmann_rel_residual":err,"floor":floor})
    except Exception as e:
        row.update({"status":"failed","error":repr(e),
                    "traceback":traceback.format_exc()[-1200:]})
    rows.append(row)
    (OUT/"planck_resonance_conditional.json").write_text(json.dumps(out,indent=2))
    print("POINT",H0,A,row.get("status"),row.get("planck_primary_lnlike"),
          row.get("error",""),flush=True)
good=[r for r in rows if r["status"]=="computed"]
if good:
    best=max(good,key=lambda r:r["planck_primary_lnlike"])
    l0=max([r for r in good if r["A_res"]==0],
           key=lambda r:r["planck_primary_lnlike"])
    out["best_grid_point"]=best
    out["best_receiver_free_point"]=l0
    out["delta_chi2_best_vs_receiver_free"]=float(-2*(
        best["planck_primary_lnlike"]-l0["planck_primary_lnlike"]))
out["status"]="complete" if len(good)==len(rows) else "partial"
(OUT/"planck_resonance_conditional.json").write_text(json.dumps(out,indent=2))
print("BEST",out.get("best_grid_point"),flush=True)
if len(good)!=len(rows):
    raise SystemExit("Some points failed; inspect partial report")
