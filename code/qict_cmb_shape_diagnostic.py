"""Native QICT CMB shape ablation with exact matched vanilla and global LCDM controls.

This code computes linear-lensing C_ell with CLASS, compares against
both the same-input LCDM background and a conventional reference.
No observational likelihood is claimed here; the measured likelihood
analysis lives in separate GitHub workflows.
"""
import json,math,traceback
from pathlib import Path
import numpy as np
from classy import Class
OUT=Path("results/qict_cmb_shape_diagnostic")
OUT.mkdir(parents=True,exist_ok=True)
OMR=2.469e-5*(1+.22710731766*3.044)
OPHI=.00038644699225749197*.674**2
SAMPLES=[100,200,300,500,800,1100,1500,2000,2500,3000]
K_SAMPLES=[.01,.02,.05,.1,.2]
CASES=[
 {"name":"reference_LCDM_67p4","kind":"lcdm","H0":67.4,"omc":.12,"ns":.965},
 {"name":"sameparams_LCDM_68p5","kind":"lcdm","H0":68.5,"omc":.121,"ns":.967},
 {"name":"QICT_68p5","kind":"qict","H0":68.5,"omc":.121,"ns":.967,"V0":36.699909254729036,"F":.1001613464888732},
 {"name":"sameparams_LCDM_70p5","kind":"lcdm","H0":70.5,"omc":.124,"ns":.975},
 {"name":"QICT_70p5","kind":"qict","H0":70.5,"omc":.124,"ns":.975,"V0":122.77781813186203,"F":.17674669285161573},
 {"name":"sameparams_LCDM_72","kind":"lcdm","H0":72.,"omc":.13,"ns":.99},
 {"name":"QICT_72","kind":"qict","H0":72.,"omc":.13,"ns":.99,"V0":235.51950122406512,"F":.23734168553381624}
]

def cosmo(p):
 d={
  "H0":p["H0"],"omega_b":.02237,"omega_cdm":p["omc"],"tau_reio":.0544,
  "A_s":2.1e-9,"n_s":p["ns"],"Omega_fld":0,
  "N_ncdm":0,"N_ur":3.044,"gauge":"newtonian",
  "output":"tCl,lCl,pCl,mPk","lensing":"yes",
  "l_max_scalars":3500,"P_k_max_1/Mpc":.4,
 }
 if p["kind"]=="lcdm":d["Omega_scf"]=0
 else:
  F=p["F"];V=p["V0"]
  d.update({
    "Omega_scf":-1,"model_cdm":"i","cdm_c":.02,
    "scf_potential":"QICT_finite_N",
    "scf_parameters":f'{V:.15g}, {F:.15g}, 250., 0., 0., 0., 0., 0., 0., 0., {2.5*F:.15g}, 0.',
    "attractor_ic_scf":"no","scf_tuning_index":0,
    "Omega_Lambda":1-(.02237+p["omc"]+OMR+OPHI)/(p["H0"]/100)**2,
  })
 return d

def run():
 rows=[]
 native={}
 for p in CASES:
  c=Class()
  row={"parameters":p}
  try:
   c.set(cosmo(p));c.compute()
   cls=c.lensed_cl(3500)
   ell=np.asarray(cls["ell"])
   spectra={name:np.asarray(cls[name],float) for name in ("tt","te","ee","bb","pp") if name in cls}
   pk={}
   for k in K_SAMPLES:
    pk[str(k)]=float(c.pk_lin(k,0))
   row.update({"status":"computed","realized_H0":c.Hubble(0)*299792.458,
    "sigma8":float(c.sigma8()),"Omega_m":float(c.Omega_m()),
    "C_ell":{"ell":SAMPLES,
      **{sp:[float(v[l]) for l in SAMPLES] for sp,v in spectra.items()}},
    "P_k_linear_z0_Mpc3":pk})
   native[p["name"]]=row
  except Exception as e:
   row.update({"status":"failed","error":repr(e),"traceback":traceback.format_exc()[-2500:]})
  finally:
   try:c.struct_cleanup();c.empty()
   except Exception:pass
  rows.append(row)
  print("QICT_CMB_SHAPE",json.dumps({k:v for k,v in row.items() if k!="C_ell"}),flush=True)
  (OUT/"cmb_shape.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
 result={"status":"partial","multipoles":SAMPLES,"models":rows,"comparisons":[]}
 base=native.get("reference_LCDM_67p4")
 for key in ("68p5","70p5","72"):
  a=native.get("QICT_"+key);b=native.get("sameparams_LCDM_"+key)
  if not (a and b and base):continue
  comp={"pair":key,"type":"QICT compared to SAME-input LCDM and reference LCDM",
        "at_ell":[]}
  for j,l in enumerate(SAMPLES):
   entry={"ell":l}
   for s in ("tt","ee"):
    v=a["C_ell"][s][j]
    ref=b["C_ell"][s][j]
    control=base["C_ell"][s][j]
    entry["relative_"+s+"_QICT_minus_matched_LCDM"]=v/ref-1 if ref!=0 else None
    entry["relative_"+s+"_QICT_minus_reference_LCDM"]=v/control-1 if control!=0 else None
   refb=b["C_ell"]["te"][j]
   entry["TE_absolute_delta_relative_to_sqrt_TT_EE"]=(a["C_ell"]["te"][j]-refb)/math.sqrt(max(1.e-100,abs(b["C_ell"]["tt"][j]*b["C_ell"]["ee"][j])))
   comp["at_ell"].append(entry)
  comp["pk_ratios"]={k:a["P_k_linear_z0_Mpc3"][k]/b["P_k_linear_z0_Mpc3"][k] for k in a["P_k_linear_z0_Mpc3"]}
  result["comparisons"].append(comp)
 result["status"]="complete" if len(native)==len(CASES) else "partial"
 (OUT/"cmb_shape.json").write_text(json.dumps(result,indent=2)+"\n")
 if result["status"]!="complete":raise SystemExit("Some native spectra missing; inspect results")


if __name__=="__main__":run()
