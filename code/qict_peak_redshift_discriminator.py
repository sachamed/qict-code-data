"""Discriminating redshift of scalar excitation at acoustically matched angle.

Based on native CLASS and official Planck 2018 + DESI DR2 likelihoods.
This is a fixed-cosmology three-by-three slice, not posterior sampling.
"""
from pathlib import Path
import qict_acoustic_anchored_shoot as stage

# At each point, independently shoot scalar V0 and F to enforce both
# CLASS 100 theta_s=1.04109 and z_peak (not a background proxy).
stage.CASES = [
    dict(name=f"H0_{h:.1f}_z{z}",H0=h,omega_cdm=om,ns=ns,
         zc=float(z),fpeak=None)
    for h,om,ns in [
        (68.5,0.1210,0.967),
        (70.0,0.1240,0.975),
        (71.0,0.1280,0.985)]
    for z in (2500,3500,5000)
]
stage.OUT=Path("results/qict_peak_epoch_scan")
stage.OUT.mkdir(parents=True,exist_ok=True)

if __name__=="__main__":
    stage.main()
