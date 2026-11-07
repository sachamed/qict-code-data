"""Physical closure and acoustic calibration audit for pinned finite-N QICT CLASS.

No likelihood claim is made here. Each separate CLASS solve is checked
against the physical input H0, full late energy budget, and recombination
acoustic scale. No guessed derived-parameter key is required for success.
"""
import json
import math
from pathlib import Path
import numpy as np
from classy import Class

OUT=Path("results/qict_friedmann_acoustic_audit")
OUT.mkdir(parents=True,exist_ok=True)
NEFF=3.044
OMEGAR=2.469e-5*(1+0.22710731766*NEFF)
OMEGAPHI0=.00038644699225749197*.674**2
PAR="201.55504807115258, 0.225, 250., 0., 0., 0., 0., 0., 0., 0., 0.5625, 0."
WEAK="2.5e-16, 0.35, 250., 0., 0., 0., 0., 0., 0., 0., 0.875, 0."


def parameters(kind,h0=74.93,g=.02):
    d={
      "H0":h0,"omega_b":.02237,"omega_cdm":.12,
      "tau_reio":.0544,"A_s":2.1e-9,"n_s":.965,
      "Omega_fld":0,"N_ncdm":0,"N_ur":NEFF,
      "gauge":"newtonian","lensing":"yes",
      "output":"tCl,pCl,lCl","l_max_scalars":3000
    }
    if kind!="lcdm":
        d.update({
          "model_cdm":"i","cdm_c":g,
          "scf_potential":"QICT_finite_N",
          "scf_parameters":WEAK if kind=="weak" else PAR,
          "attractor_ic_scf":"no","scf_tuning_index":0,
        })
        if kind=="weak":d["Omega_scf"]=1e-8
        else:
            d["Omega_scf"]=-1
            d["Omega_Lambda"]=1-(.02237+.12+OMEGAR+OMEGAPHI0)/(h0/100)**2
    else:
        d["Omega_scf"]=0
    return d


def run_one(kind,h0,g):
    p=parameters(kind,h0,g)
    c=Class()
    row={"kind":kind,"H0_input":h0,"g":g if kind!="lcdm" else None,
         "input_Omega_Lambda":p.get("Omega_Lambda")}
    try:
        c.set(p);c.compute()
        H0=c.Hubble(0)*299792.458
        row.update({"status":"computed","H0_derived_from_Hubble0_km_s_Mpc":H0,
                    "H0_fractional_deviation":H0/h0-1,
                    "rdrag_if_available":None})
        # Every candidate must satisfy the *actual* cosmological H0.
        if abs(row["H0_fractional_deviation"])>5e-4:
            row["closure_flag"]="failed_H0"
        for par in ("100*theta_s","z_rec","z_d","rs_rec","rs_d","rs_drag","Omega_m","sigma8","age"):
            try:
                x=c.get_current_derived_parameters([par])[par]
                if isinstance(x,(float,int)) and math.isfinite(float(x)):row[par]=float(x)
            except Exception as err:row.setdefault("unsupported_derived_parameters",[]).append(par)
        bg=c.get_background()
        row["background_headers"]=list(bg)
        zkey=next((name for name in bg if name.strip()=="z"),None)
        hkey=next((name for name in bg if "H [1/Mpc]" in name),None)
        if zkey and hkey:
            zz=np.asarray(bg[zkey]);HH=np.asarray(bg[hkey])
            k=np.argsort(zz); zz=zz[k];HH=HH[k]
            row["background_H_1090_inv_Mpc"]=float(np.interp(1090.,zz,HH))
            row["background_H_3500_inv_Mpc"]=float(np.interp(3500.,zz,HH))
            # Actual scalar and matter components in the CLASS output may
            # be named differently in future forks. Record original keys.
            names=list(bg)
            scf=[name for name in names if "rho_scf" in name]
            tot=[name for name in names if "rho_tot" in name]
            if scf and tot:
                rr=np.asarray(bg[scf[0]])[k];rt=np.asarray(bg[tot[0]])[k]
                for z in (0,1100,3500):
                    row[f"f_scf_actual_{z}"]=float(np.interp(z,zz,rr/rt))
                row["scalar_closure_delta_Omega0_physical"]=(row["f_scf_actual_0"]-(OMEGAPHI0/(h0/100)**2)) if kind=="high" else None
            row["background_key_rho_scf"]=scf
            row["background_key_rho_tot"]=tot
        row["Omega_m_from_classy"]=float(c.Omega_m())
        # CLASS sigma8 requires mPk output. Its absence is not a Friedmann failure.
        try: row["sigma8_from_classy"]=float(c.sigma8())
        except Exception as ex: row["sigma8_unavailable_reason"]=str(ex)
        if "100*theta_s" in row:
            row["theta_100_distance_from_planck_reference"]=row["100*theta_s"]-1.04109
        if row.get("status")=="computed" and abs(row["H0_fractional_deviation"])<=5e-4:
            row["closure_flag"]="passed_H0_only_pending_density_checks"
    except Exception as ex:
        import traceback
        row.update({"status":"failed","error":repr(ex),"traceback":traceback.format_exc()[-6000:]})
    finally:
        try:c.struct_cleanup()
        except Exception:pass
        try:c.empty()
        except Exception:pass
    print("QICT_FRIEDMANN_ACOUSTIC_AUDIT",json.dumps({k:v for k,v in row.items() if k not in ("background_headers","traceback")}),flush=True)
    return row


def main():
    tests=[("lcdm",67.4,0),("lcdm",74.93,0),
           ("high",67.4,.02),("high",74.93,.02),
           ("high",74.93,0),("weak",67.4,.02)]
    report={"status":"running","physical_checks":"H0, scalar fractions, native acoustic-derived parameters",
            "model":"native coupled CLASS, no surrogate", "points":[]}
    for x in tests:
        row=run_one(*x)
        report["points"].append(row)
        (OUT/"physical_closure.json").write_text(json.dumps(report,indent=2)+"\n")
    report["status"]="complete" if all(row["status"]=="computed" for row in report["points"]) else "partial"
    (OUT/"physical_closure.json").write_text(json.dumps(report,indent=2)+"\n")
    if report["status"]!="complete":
        raise SystemExit("One native CLASS acoustic diagnostic failed")


if __name__=="__main__":main()
