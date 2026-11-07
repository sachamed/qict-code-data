"""Second discriminating scan of finite-charge QICT peak redshift.

F and V0 are independently recomputed from full CLASS background, matching
physical zpeak and CMB acoustic angle for every point. Official Planck 2018
and DESI DR2 likelihoods are evaluated with identical data.
This is a fixed cosmological-parameter slice, not a converged model fit.
"""
from pathlib import Path
import qict_acoustic_anchored_shoot as stage

stage.CASES=[
 dict(name=f"H0_{h:.1f}_z{z}",H0=h,omega_cdm=om,ns=ns,zc=float(z),fpeak=None)
 for h,om,ns in [(70.,.124,.975),(71.,.128,.985),(72.,.13,.99)]
 for z in (1500,2000,2500,3000)
]
stage.OUT=Path("results/qict_peak_epoch_refinement")
stage.OUT.mkdir(parents=True,exist_ok=True)
if __name__=="__main__":
 stage.main()
