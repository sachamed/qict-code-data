"""Physical z_peak,f_peak shooting in native QICT CLASS, then official Planck+DESI.

An exploratory two-point profile, not a converged posterior. Every
failed background or likelihood is explicitly recorded.
"""
import json, math, time, traceback
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from classy import Class
from cobaya.model import get_model

OUT=Path("results/qict_peak_calibration")
OUT.mkdir(parents=True,exist_ok=True)
N,THETA=250,2.5
OM_B=.02237
OM_R=2.469e-5*(1+.22710731766*3.044)
TARGETS=[
  dict(name="moderate",H0=70.5,omega_cdm=.128,ns=.99,zc=3400.,fpeak=.05),
  dict(name="intermediate",H0=71.5,omega_cdm=.132,ns=.99,zc=3500.,fpeak=.07),
  dict(name="large",H0=73.04,omega_cdm=.138,ns=1.,zc=3500.,fpeak=.095),
]
LIKES={
 "planck_2018_highl_plik.TTTEEE_lite_native":None,
 "planck_2018_lowl.TT":None,
 "planck_2018_lowl.EE":None,"bao.desi_dr2":None
}
def micro(theta):
 x=1-np.cos(theta)
 a,b,c=4/N**2,6*(N-1)/N**2,(N-1)*(N-2)/N**2
 return a*x+b*x*x+c*x**3,(2*b+6*c*x)*np.sin(theta)**2+(a+2*b*x+3*c*x*x)*np.cos(theta)

def seed(t):
 P,Q=micro(THETA);assert Q<0
 h=t["H0"]/100;z=t["zc"];f=t["fpeak"]
 Hbg2=(t["H0"]/299792.458)**2*((OM_B+t["omega_cdm"])/h**2*(1+z)**3+
          OM_R/h**2*(1+z)**4+.685)
 return 3*f*Hbg2/((1-f)*P), float(np.sqrt(3*f*abs(Q)/(4*P)))

def cfg(t,V,F,standard=False):
 h=t["H0"]/100.
 d=dict(H0=t["H0"],omega_b=OM_B,omega_cdm=t["omega_cdm"],
        tau_reio=.0544,A_s=2.1e-9,n_s=t["ns"],
        Omega_fld=0,N_ncdm=0,N_ur=3.044,gauge="newtonian")
 if standard:d["Omega_scf"]=0;return d
 pars=f"{V:.15g}, {F:.15g}, {N}., 0., 0., 0., 0., 0., 0., 0., {F*THETA:.15g}, 0."
 d.update(model_cdm="i",cdm_c=.02,scf_potential="QICT_finite_N",
        scf_parameters=pars,attractor_ic_scf="no",scf_tuning_index=0,
        Omega_scf=-1,Omega_Lambda=1-(OM_B+t["omega_cdm"]+OM_R+.00017)/h**2)
 return d

def physical(t,V,F):
 c=Class()
 try:
  c.set(cfg(t,V,F));c.compute()
  b=c.get_background()
  z=np.asarray(b["z"],float);f=np.asarray(b["(.)rho_scf"],float)/np.asarray(b["(.)rho_tot"],float)
  valid=(z>=100)&(z<=10000)&np.isfinite(f)
  if not np.any(valid):raise ValueError("No finite fraction")
  inds=np.where(valid)[0];i=inds[np.argmax(f[inds])]
  zpk=float(z[i]);fpk=float(f[i])
  # Resolve the maximum between CLASS background samples; a discrete
  # argmax otherwise gives an artificially zero finite-difference Jacobian.
  if 1<=i<len(z)-1 and np.all(f[i-1:i+2]>0):
   xx=np.log1p(z[i-1:i+2]);yy=np.log(f[i-1:i+2])
   aa,bb,cc=np.polyfit(xx,yy,2)
   if aa<0:
    xm=-bb/(2*aa)
    if min(xx)<=xm<=max(xx):
     zpk=float(np.expm1(xm))
     fpk=float(np.exp(np.polyval([aa,bb,cc],xm)))
  H0=c.Hubble(0)*299792.458
  r={"peak_z":zpk,"peak_f":fpk,"H0_realized":float(H0),
     "H0_rel_error":float(H0/t["H0"]-1)}
  for k in ("rs_d","100*theta_s","z_rec"):
   try:r[k]=float(c.get_current_derived_parameters([k])[k])
   except Exception:r[k]=None
  return r
 finally:
  try:c.struct_cleanup();c.empty()
  except Exception:pass

def official(t,V,F,standard=False):
 d=cfg(t,V,F,standard)
 fixed={k:d.pop(k) for k in ("H0","omega_b","omega_cdm","tau_reio","A_s","n_s")}
 if not standard:fixed["cdm_c"]=d.pop("cdm_c")
 d.update({"non_linear":None,"lensing":"yes"})
 model=get_model({"packages_path":"/tmp/cobaya_packages","debug":False,
                  "theory":{"classy":{"stop_at_error":False,"extra_args":d}},
                  "likelihood":LIKES,"params":{**fixed,"A_planck":1.}})
 p=model.logposterior({},as_dict=True)
 likes={k:float(v) for k,v in p["loglikes"].items()}
 if len(likes)!=4 or not all(np.isfinite(v) for v in likes.values()):raise RuntimeError("Nonfinite likelihood")
 return {"loglikes":likes,"sum":float(sum(likes.values()))}

def main():
 report={"status":"running","scope":"Full native CLASS shooting followed by official Planck and DESI at three hand-selected cosmological points",
         "calibration":"two-dimensional V0,F nonlinear shoot to z_peak and f_peak; alpha=4 only initial seed",
         "not_converged_joint_fit":True,"results":[]}
 base=dict(name="lcdm",H0=67.4,omega_cdm=.12,ns=.965)
 try:
  report["lcdm_reference"]=official(base,0,.2,True)
 except Exception as e:report["lcdm_reference"]={"error":repr(e)}
 for t in TARGETS:
  row={"target":t,"background_trials":[],"status":"starting"}
  t0=time.monotonic()
  V,F=seed(t);row["seed"]={"V0":V,"F":F}
  # The alpha=4 frozen-onset approximation systematically locates the
  # actual energy peak too late. Rescale V0 by H^2(z) ~ (1+z)^3.5
  # before finite-difference shooting. This is merely a first guess.
  try:
   seed_bg=physical(t,V,F)
   row["seed_background"]=seed_bg
   V=float(np.clip(V*(t["zc"]/seed_bg["peak_z"])**3.5,5.01,2999.))
   row["redshift_rescaled_seed"]={"V0":V,"F":F}
  except Exception as exc:row["seed_rescaling_failure"]=repr(exc)
  def fun(logx):
   V,F=np.exp(logx)
   try:
    d=physical(t,V,F)
    row["background_trials"].append({"V0":float(V),"F":float(F),**d})
    r=np.array([np.log(d["peak_z"]/t["zc"]),np.log(d["peak_f"]/t["fpeak"])])
    if abs(d["H0_rel_error"])>.001:r+=2
   except Exception as e:
    row["background_trials"].append({"V0":float(V),"F":float(F),"error":repr(e)})
    r=np.array([5.,5.])
   print("QICT_PEAK_TRIAL",t["name"],float(V),float(F),r.tolist(),flush=True)
   return r
  try:
   opt=least_squares(fun,np.log([V,F]),bounds=(np.log([5,.06]),np.log([3000,.9])),
                     max_nfev=28,diff_step=.04,ftol=.003,xtol=.003,gtol=.003)
   valid=[r for r in row["background_trials"] if "peak_f" in r]
   if not valid:raise RuntimeError("No valid physical backgrounds")
   best=min(valid,key=lambda r:np.log(r["peak_z"]/t["zc"])**2+
                                        np.log(r["peak_f"]/t["fpeak"])**2)
   row["best_background"]=best;row["solver_message"]=str(opt.message)
   dz=best["peak_z"]/t["zc"]-1;df=best["peak_f"]/t["fpeak"]-1
   row["target_rel_error"]={"z":float(dz),"f":float(df)}
   if max(abs(dz),abs(df))>.10 or abs(best["H0_rel_error"])>.001:
    row["status"]="rejected_physical_peak_or_Friedmann"
   else:
    row["likelihood"]=official(t,best["V0"],best["F"])
    row["status"]="computed"
    if "sum" in report.get("lcdm_reference",{}):
     row["delta_chi2_vs_fixed_lcdm"]=-2*(row["likelihood"]["sum"]-report["lcdm_reference"]["sum"])
  except Exception as e:
   row["status"]="failed";row["error"]=repr(e);row["traceback"]=traceback.format_exc()[-4000:]
  row["elapsed_seconds"]=round(time.monotonic()-t0,2)
  report["results"].append(row)
  (OUT/"calibrated_grid.json").write_text(json.dumps(report,indent=2)+"\n")
  print("QICT_PEAK_CALIBRATED_RESULT",json.dumps({k:v for k,v in row.items() if k!="background_trials"}),flush=True)
 report["status"]="complete" if all(x["status"]=="computed" for x in report["results"]) else "partial"
 (OUT/"calibrated_grid.json").write_text(json.dumps(report,indent=2)+"\n")
 if not any(x["status"]=="computed" for x in report["results"]):raise SystemExit("No calibrated observed point")

if __name__=="__main__":main()
