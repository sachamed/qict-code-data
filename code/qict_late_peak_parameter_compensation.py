"""Observed high-H0 QICT physical late-peak parameter compensation grid.

At each H0, omega_cdm, n_s triple independently solve CLASS for scalar
V0,F from actual zpeak=2000 and CMB 100theta_s=1.04109.
Then evaluate official Planck 2018 TTTEEE+lowT+lowE and DESI DR2.
This limited conditional grid is NOT independently optimized LCDM
or an MCMC posterior.
"""
from pathlib import Path
import qict_acoustic_anchored_shoot as stage

stage.CASES=[
 dict(name=f"H0_{h:g}_w{om:.3f}_ns{ns:.3f}",H0=h,
      omega_cdm=om,ns=ns,zc=2000.,fpeak=None)
 for h,oms,nss in [
    (70.,[.122,.124,.126],[.965,.975,.985]),
    (71.,[.126,.128,.130],[.975,.985,.995])
 ]
 for om in oms for ns in nss
]
stage.OUT=Path("results/qict_late_peak_parameter_compensation")
stage.OUT.mkdir(parents=True,exist_ok=True)
if __name__=="__main__":stage.main()
