"""H0 low-end acoustic-surface continuation using the validated native shooter.

Fixes z_peak=3500 and observed 100theta_s, solves F,V0, and evaluates
Planck+DESI. A conditional slice, not a posterior or general bound.
"""
from pathlib import Path
import qict_acoustic_anchored_shoot as stage

stage.CASES=[
 dict(name="H0_68p5",H0=68.5,omega_cdm=.121,ns=.967,zc=3500.,fpeak=None),
 dict(name="H0_69p0",H0=69.0,omega_cdm=.1225,ns=.970,zc=3500.,fpeak=None),
 dict(name="H0_69p5",H0=69.5,omega_cdm=.123,ns=.972,zc=3500.,fpeak=None),
 dict(name="H0_70p0",H0=70.0,omega_cdm=.124,ns=.975,zc=3500.,fpeak=None),
]
stage.OUT=Path("results/qict_acoustic_lowh0")
stage.OUT.mkdir(parents=True,exist_ok=True)

if __name__=="__main__":
 stage.main()
