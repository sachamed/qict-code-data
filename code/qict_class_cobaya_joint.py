"""Truly coupled finite-N QICT CLASS + measured Planck/ACT lens/DESI fixed points.

The QICT CLASS fork is compiled and installed on a GitHub runner. This is
not a surrogate based on independently set scalar fluid sound speed.

The two ACT PRIMARY and Planck PRIMARY likelihoods are not naively multiplied.
The published joint ACT-Planck lensing variant replaces standalone lensing.
SH0ES is an explicit held-out Gaussian log likelihood; no posterior is claimed.
"""
import json,traceback
from pathlib import Path
import numpy as np
from classy import Class  # Fail early unless the compiled, editable CLASS extension imports
import act_dr6_lenslike as act
from cobaya.likelihood import Likelihood
from cobaya.model import get_model

class ACTDR6QICTJointLensing(Likelihood):
    """ACT-Planck combined lensing with primary-compatible corrections."""
    def initialize(self):
        self.data=act.load_data("actplanck_baseline",lens_only=False,
                            like_corrections=True)
        self.num_bands=int(len(self.data["data_binned_clkk"]))

    def get_requirements(self):
        return {"Cl":{"tt":4000,"te":4000,"ee":4000,"bb":4000,"pp":4000}}

    def logp(self, **kwargs):
        C=self.provider.get_Cl(ell_factor=False,units="FIRASmuK2")
        ell=np.asarray(C["ell"],dtype=int)
        kk=.25*(ell*(ell+1))**2 * C["pp"]
        val=act.generic_lnlike(self.data,ell,kk,ell,
                                C["tt"],C["ee"],C["te"],C["bb"])
        return float(val) if np.isfinite(val) else -np.inf

def run():
    out=Path("results/qict_native_joint_observed")
    out.mkdir(parents=True,exist_ok=True)
    lenslike="qict_class_cobaya_joint.ACTDR6QICTJointLensing"
    param_list="2.5e-16, 0.35, 250., 0., 0., 0., 0., 0., 0., 0., 0.875, 0."
    info={
      "packages_path":"/tmp/cobaya_packages",
      "debug":True,
      "theory":{"classy":{"stop_at_error":True,
        "extra_args":{
           "model_cdm":"i","scf_potential":"QICT_finite_N",
           "scf_parameters":param_list,"attractor_ic_scf":"no",
           "scf_tuning_index":0,"Omega_scf":1e-8,
           "gauge":"newtonian","nbody_gauge_transfer_functions":"no",
           "Omega_fld":0,
           "N_ncdm":1,"m_ncdm":0.06,"N_ur":2.0328,
           "non linear":"hmcode",
        }}},
      "likelihood":{
           "planck_2018_highl_plik.TTTEEE_lite_native":None,
           "planck_2018_lowl.TT":None,
           "planck_2018_lowl.EE":None,
           lenslike:None,
           "bao.desi_dr2":None,
       },
      "params":{
        "H0":67.4,"omega_b":.02237,"omega_cdm":.12,
        "tau_reio":.0544,"A_s":2.1e-9,"n_s":.965,
        "A_planck":1.,
        "cdm_c":{"prior":{"min":0.,"max":.05}},
      }
    }
    probe=Class()  # Check compiled classy extension before Cobaya discovery
    del probe
    report={"status":"starting","model":"QICT finite-N=250 canonical scalar + genuinely variable-mass CDM",
       "baseline":"g=0 vs g=.02, weak 1e-8 present scalar, other cosmological parameters fixed",
       "joint_sources":"Planck Plik-lite TTTEEE+Commander lowTT+SimAll lowEE+official joint ACT-Planck DR6 lensing+DESI DR2 BAO; local H0 Gaussian separate",
       "ACT_corrections":"lens_only=False,like_corrections=True,ACTPlanck variant",
       "ACT_primary_overlap":"not included","MCMC_converged":False,
       "m_chi_absolute_eV":None,"relic_abundance_absolute_prediction":None,
       "code":"classy from patched /tmp/qict_idm; one cosmological model at two g values",
       "results":[]}
    try:
        model=get_model(info)
        for g in [0.,.02]:
            row={"g":g}
            try:
                posterior=model.logposterior({"cdm_c":g},as_dict=True)
                logls={k:float(v) for k,v in posterior["loglikes"].items()}
                if not all(np.isfinite(v) for v in logls.values()):
                    raise ValueError("One of the official survey likelihoods is non-finite")
                h0=float(info["params"]["H0"])
                gauss=-.5*((h0-73.04)/1.04)**2
                row.update({"status":"computed","components_loglikes":logls,
                    "observed_primary_lensing_BAO_lnlike":float(sum(logls.values())),
                    "SH0ES_loglike_no_norm":gauss,
                    "all_plus_local_H0_lnlike":float(sum(logls.values())+gauss)})
            except Exception as e:
                row.update({"status":"failed","error":repr(e),
                            "traceback":traceback.format_exc()[-9000:]})
            report["results"].append(row)
            (out/"qict_native_joint_observed.json").write_text(json.dumps(report,indent=2))
            print("QICT_NATIVE_OBSERVED",json.dumps(row),flush=True)
    except Exception as e:
        report["initialization_failure"]=repr(e)
        report["traceback"]=traceback.format_exc()[-9000:]
    report["status"]="complete" if len(report["results"])==2 and all(r["status"]=="computed" for r in report["results"]) else "failed"
    (out/"qict_native_joint_observed.json").write_text(json.dumps(report,indent=2))
    if report["status"]!="complete":
        raise SystemExit("Observed QICT CLASS run incomplete, inspect JSON")
if __name__=="__main__":run()
