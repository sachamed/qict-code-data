"""Matched fixed-input QICT model ablations: finite charge, cubic limit, DM coupling.

Official Planck 2018 Plik-lite TTTEEE+lowTT+lowEE and DESI DR2 BAO;
identical H0, physical densities, primordial spectra and bare scalar
parameters. Tests potential shape and coupling separately, NOT independent
fully fitted models. N=1e6 approximates cubic potential; it is not claimed
to be the operator-level factorial parent Hamiltonian.
"""
from pathlib import Path
import json
import traceback
import numpy as np
from cobaya.model import get_model

OUT=Path("results/qict_physics_ablation")
OUT.mkdir(parents=True,exist_ok=True)
H0=74.93
PHI0=0.5625
F=0.225
V0=201.55504807115258
NEFF=3.044
OMEGA_R=2.469e-5*(1+.22710731766*NEFF)
# An input held fixed across these ablations. Not automatically a
# self-consistent physical scalar fraction for every charge and coupling.
OMEGA_PHI_REFERENCE=.00038644699225749197*.674**2
OMEGA_LAMBDA=1-(.02237+.12+OMEGA_R+OMEGA_PHI_REFERENCE)/(.7493**2)
LIKES={
    "planck_2018_highl_plik.TTTEEE_lite_native":None,
    "planck_2018_lowl.TT":None,
    "planck_2018_lowl.EE":None,
    "bao.desi_dr2":None
}
BASEPARAMS={
    "H0":H0,"omega_b":.02237,"omega_cdm":.12,
    "tau_reio":.0544,"A_s":2.1e-9,"n_s":.965,"A_planck":1.
}
POINTS=[
    ("finite_N250_g002",250,0.02),
    ("finite_N250_g0",250,0.0),
    ("cubic_limit_N1000000_g002",1000000,0.02),
    ("cubic_limit_N1000000_g0",1000000,0.0),
    ("standard_LCDM",None,None)
]


def one(label,N,g):
    if N is None:
        extra={"Omega_scf":0,"Omega_fld":0,"gauge":"newtonian",
            "non_linear":None,"N_ncdm":0,"N_ur":NEFF,"lensing":"yes"}
    else:
        pars=f"{V0}, {F}, {float(N)}, 0., 0., 0., 0., 0., 0., 0., {PHI0}, 0."
        extra={"model_cdm":"i","scf_potential":"QICT_finite_N",
            "scf_parameters":pars,"attractor_ic_scf":"no",
            "scf_tuning_index":0,"Omega_scf":-1,
            "Omega_Lambda":OMEGA_LAMBDA,"Omega_fld":0,
            "gauge":"newtonian","non_linear":None,
            "N_ncdm":0,"N_ur":NEFF,"lensing":"yes"}
    params=dict(BASEPARAMS)
    if g is not None: params["cdm_c"]=g
    info={"packages_path":"/tmp/cobaya_packages","debug":False,
        "theory":{"classy":{"stop_at_error":False,"extra_args":extra}},
        "likelihood":LIKES,"params":params}
    row={"name":label,"N":N,"g":g,"H0":H0,
         "Omega_Lambda_input":OMEGA_LAMBDA if N is not None else None}
    try:
        model=get_model(info)
        post=model.logposterior({},as_dict=True)
        logs={str(k):float(v) for k,v in post["loglikes"].items()}
        if len(logs)!=4 or not all(np.isfinite(v) for v in logs.values()):
            raise ArithmeticError("missing or nonfinite official loglike")
        row.update({"status":"computed","loglikes":logs,"total":float(sum(logs.values()))})
    except Exception as e:
        row.update({"status":"failed","error":repr(e),"traceback":traceback.format_exc()[-6000:]})
    print("QICT_ABLATION",json.dumps(row),flush=True)
    return row


def main():
    result={"status":"running","model_relation":"same phenomenological potential with finite-N or N=1e6 coefficient, not microscopic Hamiltonian equivalence",
        "likelihoods":list(LIKES),
        "type":"fixed-input ablation, not MCMC or optimized comparison",
        "nonlinear":"off","parameter_control":"same physical densities, H0, A_s, n_s, tau, phi_ini, F, V0, Omega_Lambda input for QICT",
        "points":[]}
    for label,n,g in POINTS:
        result["points"].append(one(label,n,g))
        (OUT/"ablation_results.json").write_text(json.dumps(result,indent=2)+"\n")
    computed={p["name"]:p for p in result["points"] if p["status"]=="computed"}
    if all(name in computed for name,_,_ in POINTS):
        base=computed["finite_N250_g002"]["total"]
        result["deltas_minus_finiteN250_g002"]={
            name:float((-2*(p["total"]-base))) for name,p in computed.items()}
        result["status"]="complete"
    else:result["status"]="incomplete"
    (OUT/"ablation_results.json").write_text(json.dumps(result,indent=2)+"\n")
    if result["status"]!="complete":
        raise SystemExit("QICT ablation incomplete; inspect per-point failures")


if __name__=="__main__":
    main()
