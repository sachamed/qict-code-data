"""Referee correction of QICT ACT DR6 lensing combinations (7 Nov 2026).

Official ACT DR6 README: lens_only=False, like_corrections=True when any
primary CMB likelihood is used. CAMB lensing needs nonlinear HMCode mead2016.
ACT-only and joint ACT+Planck LENSING products are alternatives.
Published QICT results using lens_only=True when adding Planck primary must
not be cited as the corrected joint primary+lensing statistic.
"""
import json
import numpy as np
from pathlib import Path
from time import time

import camb
from camb import model as camb_model
import act_dr6_lenslike as act
from qict_cmb_massive_boltzmann import history

OUT=Path("results/qict_corrected_act_dr6")
OUT.mkdir(parents=True,exist_ok=True)
points=(0.0,0.01,0.03,0.03776)
variants=("act_baseline","actplanck_baseline")
data_corr={variant:act.load_data(variant,lens_only=False,like_corrections=True)
           for variant in variants}
data_lensonly={variant:act.load_data(variant,lens_only=True,like_corrections=False)
               for variant in variants}
report={"date":"2026-11-07",
        "theory":"QICT one-pole uncoupled reconstructed scalar; fixed 6 vanilla parameters",
        "parameter_settings":{"H0":69.28,"ombh2":0.02237,"omch2":0.119,
          "mnu_eV":0.06,"nnu":3.044,"tau":0.054,"As":2.1e-9,"ns":0.965,
          "Hc_over_H0":1e5},
        "camb_accuracy":{"lmax":4000,"lens_output_margin":1250,
                          "lens_potential_accuracy":4,
                          "nonlinear":"mead2016"},
        "official_act_configuration_for_combination":
             "lens_only=False; like_corrections=True",
        "lensing_only_diagnostic_configuration":
             "lens_only=True; like_corrections=False",
        "overlap":"actplanck_baseline replaces standalone ACT and Planck lensing",
        "Planck_primary_loglike":None,
        "posterior_or_marginalization":None,
        "results":[]}
for lam in points:
    start=time()
    a,w,E,resid,onu=history(lam=lam)
    p=camb.CAMBparams()
    p.set_cosmology(H0=69.28,ombh2=.02237,omch2=.119,
                    mnu=.06,nnu=3.044,num_massive_neutrinos=1,tau=.054)
    p.InitPower.set_params(As=2.1e-9,ns=.965)
    if lam==0: p.set_dark_energy(w=-1,cs2=1,dark_energy_model="fluid")
    else: p.set_dark_energy_w_a(a,w,dark_energy_model="fluid")
    p.set_for_lmax(4000,lens_potential_accuracy=4,
                   lens_output_margin=1250,nonlinear=True)
    p.NonLinearModel.set_params(halofit_version="mead2016")
    # Set both so that lensing and nonlinear power use the same prescription.
    p.NonLinear=camb_model.NonLinear_both
    r=camb.get_results(p)
    cls=r.get_cmb_power_spectra(p,lmax=4000,CMB_unit="muK",raw_cl=True)["total"]
    pp=r.get_lens_potential_cls(lmax=4000,raw_cl=True)[:,0]
    ell=np.arange(len(pp))
    kk=.25*(ell*(ell+1))**2*pp
    row={"lambda":lam,"camb_version":camb.__version__,
         "Friedmann_rel_max":float(resid),
         "derived_parameters":r.get_derived_params(),
         "seconds":round(time()-start,2)}
    for variant in variants:
        corrected=float(act.generic_lnlike(data_corr[variant],ell,kk,ell,
                      cls[:,0],cls[:,1],cls[:,3],cls[:,2]))
        lens_only=float(act.generic_lnlike(data_lensonly[variant],ell,kk,ell,
                      cls[:,0],cls[:,1],cls[:,3],cls[:,2]))
        row[variant]={"lnlike_primary_compatible":corrected,
                      "chi2_primary_compatible":-2*corrected,
                      "lnlike_lensing_only_diagnostic":lens_only,
                      "chi2_lensing_only_diagnostic":-2*lens_only,
                      "n_bands":len(data_corr[variant]["data_binned_clkk"])}
    report["results"].append(row)
    (OUT/"corrected_ACT_DR6_lensing_results.json").write_text(json.dumps(report,indent=2,default=float))
    print("COMPLETE",lam,row["act_baseline"]["chi2_primary_compatible"],row["actplanck_baseline"]["chi2_primary_compatible"],flush=True)
np.savetxt(OUT/"corrected_ACT_DR6_lensing_table.csv",
 [[z["lambda"],z["act_baseline"]["chi2_primary_compatible"],
   z["actplanck_baseline"]["chi2_primary_compatible"],
   z["act_baseline"]["chi2_lensing_only_diagnostic"],
   z["actplanck_baseline"]["chi2_lensing_only_diagnostic"]] for z in report["results"]],
 delimiter=",",header="lambda,ACT_primary_compatible_chi2,ACTPlanck_primary_compatible_chi2,ACT_lens_only_chi2,ACTPlanck_lens_only_chi2",comments="")
