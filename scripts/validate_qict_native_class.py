"""Reject any QICT CLASS output that violates mass-number conservation or has bad spectra.

Loads native CLASS dimensionless D_ell and validates both g=0 and g=.02
with the exact same background and cosmological hyperparameters.
The source format is the stock CLASS _cl_lensed.dat,_cl.dat,_pk.dat and
_background.dat as generated from the patched kabeleh/iDM fork.
"""
from pathlib import Path
import json, numpy as np
base=Path("results/qict_native_class")
spectrum={}
for g in ("0.00","0.02"):
  tag=f"qict_class_g{g}_00"
  bg=np.loadtxt(base/(tag+"_background.dat"))
  cl=np.loadtxt(base/(tag+"_cl_lensed.dat"))
  unl=np.loadtxt(base/(tag+"_cl.dat"))
  pk=np.loadtxt(base/(tag+"_pk.dat"))
  assert bg.ndim==2 and cl.ndim==2 and pk.ndim==2
  assert cl.shape[1]>=6 and unl.shape[1]>=6
  assert np.all(np.isfinite(cl)) and np.all(np.isfinite(unl)) and np.all(np.isfinite(bg))
  assert np.all(np.diff(cl[:,0])==1) and int(cl[0,0])==2
  assert np.all(cl[:,[1,2,4,5]]>=0) # TT, EE, BB, phi-phi spectra
  assert np.all(np.diff(pk[:,0])>0) and np.all(pk[:,1]>0)
  z=bg[:,0]; aa=1.0/(1.0+z); phi=bg[:,17];rhoc=bg[:,10]
  mf=np.sqrt(1.0+float(g)*np.sin(phi/(2*.35))**2)
  invariant=rhoc*aa**3/mf
  mask=(z>=0)&(z<1e8)
  rel=np.max(abs(invariant[mask]/np.median(invariant[mask])-1))
  if rel>=1e-8:raise AssertionError(f"QICT cold particle number drift {rel}")
  spectrum[g]={"lmax_lensed":int(cl[-1,0]),"lmax_unlensed":int(unl[-1,0]),
     "cmb_rows":int(cl.shape[0]),"matter_power_rows":int(pk.shape[0]),
     "number_current_relative_drift":float(rel),
     "phi_early":float(phi[0]),"phi_today":float(phi[-1]),
     "m_early_over_today":float(mf[0]/mf[-1]),
     "scalar_energy_today":float(bg[-1,14]),
     "spectra_are_finite":True}
for key,idx in [("TT",1),("EE",2),("TE",3),("phiphi",5)]:
  def read(g):return np.loadtxt(base/(f"qict_class_g{g}_00_cl_lensed.dat"))
  a,b=read("0.00"),read("0.02")
  assert np.array_equal(a[:,0],b[:,0])
  mask=a[:,0]>=30
  norm=np.sqrt(np.mean(a[mask,idx]**2))
  delta=np.sqrt(np.mean((a[mask,idx]-b[mask,idx])**2))
  spectrum["delta_rms_"+key]=float(delta/norm)
out={"status":"verified CLASS QICT coupled smoke, NOT full observational validation",
     "upstream_CLASS_fork":"kabeleh/iDM GPL-3.0",
     "settings":{"N":250,"F_over_Mpl":.35,"g_values":[0,.02],
         "Omega_scf_today":1e-8,"scf_potential":"finite-charge coherent-state potential"},
     "results":spectrum}
(base/"verified_native_spectra.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
