"""Independent official ACT DR6 primary-CMB diagnostic for the calibrated H0=70 QICT candidate.

ACT foreground-marginalized primary TT/TE/EE only. Do not add to a full
Planck Plik-lite likelihood without an overlap-controlled covariance.
"""
from pathlib import Path
import qict_act_primary_native_qict as stage

stage.OUT=Path("results/qict_act_primary_calibrated_H0_70")
stage.OUT.mkdir(parents=True,exist_ok=True)
stage.POINTS=[
 dict(name="LCDM_reference_H0_67p4",kind="lcdm",H0=67.4,omega_cdm=.12,n_s=.965),
 dict(name="LCDM_same_H0_70",kind="lcdm",H0=70.,omega_cdm=.122,n_s=.975),
 dict(name="QICT_calibrated_H0_70",kind="qict",H0=70.,omega_cdm=.122,n_s=.975,
      V0=11.00848689636948,F=.14332514879730884),
 dict(name="QICT_acoustic_H0_68p5",kind="qict",H0=68.5,omega_cdm=.121,n_s=.967,
      V0=36.699909254729036,F=.1001613464888732)
]
if __name__=="__main__":stage.main()
