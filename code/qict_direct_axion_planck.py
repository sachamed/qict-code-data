"""Direct-field Einstein-Boltzmann test for the QICT-compatible cubic rotor.

IMPORTANT: The continuum potential V=V0(1-cos(phi/F))^3 is identical to
a standard n=3 axionlike early-dark-energy potential, already implemented
as CAMB EarlyQuintessence. We use CAMB's actual scalar perturbation equations,
not an imposed dark-energy fluid or a QICT-specific Boltzmann modification.

The finite receiver Hamiltonian does not fix (F,V0,theta_i) or the gravitational
matching; zc and fde_zc here are explicitly calibrated phenomenological inputs.
"""
from pathlib import Path
import json, traceback
import numpy as np
import camb
from camb import dark_energy
from cobaya.theory import Theory
from cobaya.model import get_model

class QICTDirectRotor(Theory):
    params={"H0":None,"ombh2":None,"omch2":None,"tau":None,
            "logA":None,"ns":None,"mnu":None,"zc":None,
            "fde_zc":None,"theta_i":None}
    lmax=3500
    def initialize(self):self._lmax=int(self.lmax)
    def must_provide(self,**requirements):
        req=requirements.get("Cl",{})
        if isinstance(req,dict) and req:
            self._lmax=max(self._lmax,max(map(int,req.values())))
        return {}
    def calculate(self,state,want_derived=True,**p):
        cp=camb.CAMBparams()
        de=dark_energy.EarlyQuintessence(
            n=3.0,theta_i=float(p["theta_i"]),use_zc=True,
            zc=float(p["zc"]),fde_zc=float(p["fde_zc"]),
            # these two values only seed CAMB's internal inversion
            f=.05,m=8e-53,frac_lambda0=1.,
            npoints=1000,min_steps_per_osc=12)
        cp.DarkEnergy=de
        cp.set_cosmology(H0=float(p["H0"]),ombh2=float(p["ombh2"]),
                         omch2=float(p["omch2"]),tau=float(p["tau"]),
                         mnu=float(p["mnu"]),nnu=3.044,
                         num_massive_neutrinos=int(float(p["mnu"])>0))
        cp.InitPower.set_params(As=float(np.exp(p["logA"])*1e-10),
                                 ns=float(p["ns"]))
        cp.set_for_lmax(self._lmax,lens_potential_accuracy=3,
                        lens_output_margin=1000,nonlinear=True)
        cp.NonLinearModel.set_params(halofit_version="mead2016")
        cp.NonLinear=camb.model.NonLinear_both
        result=camb.get_results(cp)
        cl=result.get_cmb_power_spectra(cp,CMB_unit="muK",raw_cl=True,
                                        lmax=self._lmax)["total"]
        pp=result.get_lens_potential_cls(lmax=self._lmax,raw_cl=True)[:,0]
        state["Cl"]={"ell":np.arange(pp.size),"tt":cl[:,0],
                     "ee":cl[:,1],"bb":cl[:,2],"te":cl[:,3],"pp":pp}
        state["derived"]={}
        # EarlyQuintessence's use_zc solver may update the parameters stored
        # in the returned result, not in the input CAMBparams. Record BOTH.
        output_de=result.Params.DarkEnergy
        zs=np.array([1000.,2500.,3400.,4000.,6500.])
        state["micro_match"]={"n":3,"theta_i":float(p["theta_i"]),
           "f_input_seed":float(cp.DarkEnergy.f),
           "m_input_seed":float(cp.DarkEnergy.m),
           "f_result":float(output_de.f),
           "m_result":float(output_de.m),
           "omega_de_z":{str(int(z)):float(result.get_Omega("de",z=float(z))) for z in zs},
           "rdrag":float(result.get_derived_params()["rdrag"]),
           "thetastar":float(result.get_derived_params().get("thetastar",np.nan))}
        return True
    def get_Cl(self,ell_factor=False,units="FIRASmuK2"):
        if units not in ("FIRASmuK2","muK2"):raise ValueError(units)
        cl=self.current_state["Cl"]
        if not ell_factor:return cl
        ll=cl["ell"].astype(float)
        out={"ell":cl["ell"]}
        f=ll*(ll+1)/(2*np.pi)
        for key in ("tt","ee","te","bb"):out[key]=cl[key]*f
        out["pp"]=cl["pp"]*(ll*(ll+1))**2/(2*np.pi)
        return out

def main():
    out=Path("results/qict_direct_scalar_planck");out.mkdir(parents=True,exist_ok=True)
    names=["planck_2018_highl_plik.TTTEEE_lite_native",
           "planck_2018_lowl.TT","planck_2018_lowl.EE"]
    fixed={"ombh2":.02237,"omch2":.12,"tau":.054,
           "logA":float(np.log(21.)),"ns":.965,"mnu":.06,
           "theta_i":2.5,"A_planck":1.}
    # zero amplitude uses another CAMB theory: physical LambdaCDM baseline
    from qict_planck_native_fixed import QICTFixedTheory
    bmodel=get_model({
        "packages_path":"/tmp/cobaya_packages",
        "theory":{"qict_planck_native_fixed.QICTFixedTheory":{"lmax":3500}},
        "likelihood":{name:None for name in names},
        "params":{**{k:v for k,v in fixed.items() if k!="theta_i"},
                  "H0":67.7,"lam":0.}},
        debug=False)
    base=bmodel.logposterior({},as_dict=True)
    baseline={name:float(v) for name,v in base["loglikes"].items()}
    if not np.all(np.isfinite(list(baseline.values()))):
        raise RuntimeError("Invalid Planck LambdaCDM baseline")
    baseline_total=float(sum(baseline.values()))
    model=get_model({
       "packages_path":"/tmp/cobaya_packages",
       "theory":{"qict_direct_axion_planck.QICTDirectRotor":{"lmax":3500}},
       "likelihood":{name:None for name in names},
       "params":{**fixed,
          "H0":{"prior":{"min":62,"max":78}},
          "zc":{"prior":{"min":2000,"max":6500}},
          "fde_zc":{"prior":{"min":0.001,"max":.3}}}})
    cases=[]
    for zc in [3400.,3000.,4000.]:
       for H0,fde in [(70.,.05),(71.5,.1),(73.,.1)]:
          cases.append((H0,zc,fde))
    report={"scientific_status":"conditional measured-data grid, not a posterior",
        "potential":"V0 [1-cos(phi/F)]^3; existing axion-like EDE n=3",
        "background_and_perturbations":"full CAMB EarlyQuintessence scalar",
        "Planck_likelihoods":names,
        "parameters_held_fixed":fixed,"baseline_LCDM":{"H0":67.7,"components":baseline,
            "total_lnlike":baseline_total},"results":[],
        "statistical_limitation":"only 3 cosmological parameters varied discretely; no nuisance refit or evidence",
        "literature_novelty":"this potential and the direct perturbation solver predate QICT"}
    for H0,zc,fde in cases:
       point={"H0":H0,"zc":zc,"fde_zc":fde}
       try:
           p=model.logposterior({"H0":H0,"zc":zc,"fde_zc":fde},as_dict=True)
           L={k:float(v) for k,v in p["loglikes"].items()}
           if not np.all(np.isfinite(list(L.values()))):
               raise ValueError("non-finite Planck TT/TE/EE likelihood")
           point.update({"status":"computed","components":L,
               "total_lnlike":float(sum(L.values())),
               "delta_chi2_vs_LCDM_control":float(-2*(sum(L.values())-baseline_total))})
           # extract field matching from the last cached likelihood state
           th=model.theory["qict_direct_axion_planck.QICTDirectRotor"]
           point["field_parameters"]=th.current_state.get("micro_match",{})
       except Exception as e:
           point.update({"status":"failed","error":repr(e),
                         "traceback":traceback.format_exc()[-2600:]})
       report["results"].append(point)
       (out/"direct_rotor_planck_conditional.json").write_text(json.dumps(report,indent=2,default=float))
       print("FIELD_QICT",point,flush=True)
    good=[z for z in report["results"] if z.get("status")=="computed"]
    if good:
        report["best_grid_point"]=max(good,key=lambda x:x["total_lnlike"])
    report["run_status"]="complete" if len(good)==len(cases) else "partial"
    (out/"direct_rotor_planck_conditional.json").write_text(json.dumps(report,indent=2,default=float))
    if report["run_status"]!="complete":
        raise SystemExit("Some direct-field test points failed, see artifact")

if __name__=="__main__":
    main()
