"""Limited independent maximum-likelihood profiling for QICT and standard LCDM.

Fixed-potential finite-charge QICT at N=250, g=0.02 versus conventional
unmodified-CDM control, each with independently varied {H0, omega_b,
omega_cdm, n_s, log(1e10 A_s)}. tau_reio=.0544 is fixed. Official
Planck 2018 Plik-lite+lowTT+lowEE and DESI DR2 likelihoods are evaluated
with the same patched CLASS build and Cobaya. SH0ES is held out.
This is a deliberately bounded numerical profile, NOT a converged fit
unless independently confirmed by convergence and multi-start tests.
"""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from cobaya.model import get_model
from scipy.optimize import minimize

OUT = Path("results/qict_h0_profile")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "evaluations.jsonl"
SUMMARY = OUT / "summary.json"
OMEGA_R = 2.469e-5 * (1.0 + 0.22710731766 * 3.044)
OMEGA_PHI_PHYSICAL = 0.00038644699225749197 * (0.674 ** 2)
POTENTIAL = "201.55504807115258, 0.225, 250., 0., 0., 0., 0., 0., 0., 0., 0.5625, 0."
BOUNDS = [(62.0, 80.0), (0.0200, 0.0250), (0.095, 0.155), (0.91, 1.03), (2.80, 3.30)]
FIELDS = ["H0", "omega_b", "omega_cdm", "n_s", "ln10_10_A_s"]
MAXFEV = int(os.environ.get("QICT_MAXFEV", "65"))


def config(kind, params):
    h0, omega_b, omega_cdm, ns, loga = [float(i) for i in params]
    extra = {
        "Omega_scf": 0,
        "Omega_fld": 0,
        "gauge": "newtonian",
        "non_linear": None,
        "N_ncdm": 0,
        "N_ur": 3.044,
        "lensing": "yes",
    }
    density = None
    if kind == "qict":
        density = 1.0 - (omega_b + omega_cdm + OMEGA_R + OMEGA_PHI_PHYSICAL) / (h0 / 100.0) ** 2
        extra.update({
            "model_cdm": "i",
            "scf_potential": "QICT_finite_N",
            "scf_parameters": POTENTIAL,
            "attractor_ic_scf": "no",
            "scf_tuning_index": 0,
            "Omega_scf": -1,
            "Omega_Lambda": density,
        })
    info = {
        "packages_path": "/tmp/cobaya_packages",
        "debug": False,
        "theory": {"classy": {"stop_at_error": False, "extra_args": extra}},
        "likelihood": {
            "planck_2018_highl_plik.TTTEEE_lite_native": None,
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "bao.desi_dr2": None,
        },
        "params": {
            "H0": h0,
            "omega_b": omega_b,
            "omega_cdm": omega_cdm,
            "n_s": ns,
            "A_s": np.exp(loga) * 1.0e-10,
            "tau_reio": 0.0544,
            "A_planck": 1.0,
        },
    }
    if kind == "qict":
        info["params"]["cdm_c"] = 0.02
    return info, density


def main():
    best = {"lcdm": None, "qict": None}
    starts = {
        "lcdm": [67.4, 0.02237, 0.12, 0.965, np.log(2.1e-9 * 1.0e10)],
        "qict": [74.93, 0.02237, 0.12, 0.965, np.log(2.1e-9 * 1.0e10)],
    }
    report = {
        "data": "official Planck 2018 Plik-lite TTTEEE, lowTT, lowEE, DESI DR2 BAO",
        "same_theory_implementation": "patched kabeleh/iDM CLASS from existing QICT workflow",
        "branch": "qict/h0-profile-2026-11-07",
        "optimizer": "bounded Powell, one initial point per model",
        "maxfev_per_model": MAXFEV,
        "varied_parameters": FIELDS,
        "fixed_parameters": {"tau_reio": 0.0544, "N_ur": 3.044, "N_ncdm": 0},
        "qict": "finite N=250, g=0.02, scalar parameters fixed; Omega_Lambda budget estimated at each point",
        "nonlinear": "OFF; ACT lensing not included",
        "posterior": "NOT sampled; best finite point is not a converged MCMC inference",
        "results": {},
    }
    for kind in ("lcdm", "qict"):
        count = 0
        t0 = time.monotonic()

        def objective(x):
            nonlocal count
            count += 1
            params = dict(zip(FIELDS, map(float, x)))
            row = {"kind": kind, "n": count, "parameters": params}
            try:
                if any(float(v) < low or float(v) > high for v, (low, high) in zip(x, BOUNDS)):
                    raise ValueError("out_of_bounds")
                info, density = config(kind, x)
                row["Omega_Lambda_input"] = density
                model = get_model(info)
                post = model.logposterior({}, as_dict=True)
                likes = {str(k): float(v) for k, v in post["loglikes"].items()}
                if len(likes) != 4 or not all(np.isfinite(v) for v in likes.values()):
                    raise ArithmeticError("non-finite or missing likelihood")
                ll = float(sum(likes.values()))
                row.update({"status": "computed", "loglikes": likes, "total": ll})
                if best[kind] is None or ll > best[kind]["total"]:
                    best[kind] = dict(row)
            except Exception as exc:
                row.update({"status": "failed", "error": repr(exc)})
                ll = -1e12
            with LOG.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print("QICT_PROFILE_EVAL", json.dumps({"kind":kind,"n":count,"H0":params["H0"],"loglike":ll,"status":row["status"]}), flush=True)
            return -2.0 * ll

        x0 = np.asarray(starts[kind], dtype=float)
        objective(x0)
        result = minimize(
            objective, x0, method="Powell", bounds=BOUNDS,
            options={"maxfev": MAXFEV, "maxiter": 16, "xtol": 0.025, "ftol": 0.002},
        )
        report["results"][kind] = {
            "evaluations": count,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "best_finite_evaluated_point": best[kind],
        }
        SUMMARY.write_text(json.dumps(report, indent=2) + "\n")
        print("QICT_PROFILE_BEST", json.dumps(report["results"][kind]), flush=True)
    if best["lcdm"] and best["qict"]:
        report["delta_chi2_QICT_minus_LCDM_at_recorded_best_points"] = -2.0 * (
            best["qict"]["total"] - best["lcdm"]["total"]
        )
        report["SH0ES_heldout_loglike_at_best_points"] = {
            kind: -0.5 * ((best[kind]["parameters"]["H0"] - 73.04) / 1.04) ** 2
            for kind in ("lcdm", "qict")
        }
    report["completed_all_models"] = all(best[x] is not None for x in ("qict", "lcdm"))
    SUMMARY.write_text(json.dumps(report, indent=2) + "\n")
    if not report["completed_all_models"]:
        raise SystemExit("No finite comparison possible; inspect evaluations.jsonl")


if __name__ == "__main__":
    main()
