"""Early-energy CLASS preflight for the QICT finite-N potential.
This is a numerical forward model and NOT a cosmological likelihood or H0 fit.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np

BRANCHES = {
    "0.00": (201.55504807115258, 0.0003940231249170709, 0.6861142701611976),
    "0.02": (201.55504807115258, 0.00036879670990630508, 0.6861394965797705),
}
F = 0.225
N = 250
THETA_INITIAL = 2.5

def update(src, key, value):
    pat = re.compile(r"(?m)^(\s*)" + re.escape(key) + r"(\s*=\s*)([^#\n]*)(.*)$")
    match = pat.search(src)
    if not match:
        return src + "\n" + key + " = " + str(value) + "\n"
    return src[:match.start()] + match.group(1) + key + match.group(2) + str(value) + match.group(4) + src[match.end():]

def prepare(class_root):
    # Start with the fork's own full input file, not a guessed minimal file.
    base = (class_root / "ini/iDM.ini").read_text()
    for g, (v0, omega_today, lambda_floor) in BRANCHES.items():
        params = f"{v0:.17g}, {F}, {N}, 0, 0, 0, 0, 0, 0, 0, {F*THETA_INITIAL:.17g}, 0"
        value_map = {
            "h":0.674,"omega_b":0.02237,"omega_cdm":0.12,
            "N_ncdm":0,"N_ur":3.044,"model_cdm":"i","cdm_c":g,
            "scf_potential":"QICT_finite_N","scf_parameters":params,
            "Omega_scf":-1,"Omega_Lambda":f"{lambda_floor:.17g}","Omega_fld":0,
            "attractor_ic_scf":"no","scf_tuning_index":0,
            "tol_initial_Omega_r":"1e-0","a_ini_over_a_today_default":"1e-20",
            "gauge":"newtonian","nbody_gauge_transfer_functions":"no",
            "output":"tCl,pCl,lCl,mPk","l_max_scalars":4000,
            "l_max_lss":120,"lensing":"yes","write_background":"yes",
            "root":f"results/qict_early_g{g}_",
        }
        s = base
        for key, val in value_map.items():
            s = update(s,key,val)
        s = re.sub(r"(?m)^\s*H0\s*=.*$", "# H0 set through h",s)
        dest = class_root / f"qict_early_g{g}.ini"
        dest.write_text(s)
        print("candidate",str(dest),"finite_charge_N",N,"g",g,"Omega_scf(today)_diagnostic",omega_today,flush=True)

def validate(root):
    result = {
        "status":"complete","kind":"finite-N CLASS Einstein-Boltzmann forward evaluation, not survey fit",
        "parameters":{"N":N,"F_over_reduced_Mpl":F,"theta_initial":THETA_INITIAL,"H0_input":67.4},
        "branches":{}
    }
    for g in BRANCHES:
        prefix = root / f"qict_early_g{g}_00_"
        files = {k:Path(str(prefix)+v) for k,v in {
            "background":"background.dat","lensed":"cl_lensed.dat","matter":"pk.dat"}.items()}
        for p in files.values():
            if not p.is_file():
                raise FileNotFoundError(str(p))
        bg=np.loadtxt(files["background"]); cl=np.loadtxt(files["lensed"]); pk=np.loadtxt(files["matter"])
        if bg.ndim != 2 or bg.shape[1] < 32 or cl.ndim != 2 or cl.shape[1] < 6 or pk.ndim != 2:
            raise AssertionError("missing columns or non-tabular output")
        if any(not np.isfinite(x).all() for x in [bg,cl,pk]):
            raise AssertionError("non-finite native spectra or densities")
        if len(cl) < 3990 or cl[0,0] != 2 or cl[-1,0] < 3999:
            raise AssertionError("insufficient lensed multipoles")
        if np.any(cl[:,[1,2,4,5]] < 0) or np.any(pk[:,1] <= 0):
            raise AssertionError("negative auto-spectrum or matter power")
        z=bg[:,0]; a=1./(1.+z)
        frac=bg[:,14]/bg[:,13]
        if not np.isfinite(frac).all() or np.any(frac < -1.e-7):
            raise AssertionError("unphysical scalar fraction")
        peak_idx=int(np.argmax(frac)); peak=float(frac[peak_idx]); zpeak=float(z[peak_idx])
        field=bg[:,17]; mass=np.sqrt(1+float(g)*np.sin(field/(2*F))**2)
        invariant=bg[:,10]*a**3/mass
        mask=np.isfinite(invariant)&(z<1.e8)&(z>=0)
        if np.count_nonzero(mask)<20:
            raise AssertionError("number current diagnostic lacks valid samples")
        drift=float(np.max(abs(invariant[mask]/np.median(invariant[mask])-1)))
        if drift>1e-7:
            raise AssertionError(f"number-current violation {g}: {drift}")
        result["branches"][g]={
          "f_peak":peak,"z_peak":zpeak,"f_z1100":float(np.interp(1100,z[::-1],frac[::-1])),
          "f_today":float(frac[np.argmin(z)]),"number_current_max_relative_drift":drift,
          "lensed_multipoles":int(len(cl)),"matter_power_samples":int(len(pk))}
        if not (0.05 < peak < 0.25 and 1700 < zpeak < 6500):
            result["status"]="failed_energy_location"
    (root/"QICT_NATIVE_EARLY_ENERGY_REPORT.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    if result["status"]!="complete":
        raise SystemExit("CLASS ran but failed physical early-energy gate")
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("prepare","validate"))
    ap.add_argument("--class-root",default="/tmp/qict_idm")
    ap.add_argument("--results",default="results/qict_native_early_energy")
    args=ap.parse_args()
    prepare(Path(args.class_root)) if args.mode=="prepare" else validate(Path(args.results))
