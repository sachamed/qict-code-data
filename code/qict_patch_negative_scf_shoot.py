"""Correct interacting CLASS scalar-fraction semantics for negative Omega_scf.
A negative scalar fraction denotes budget closure, not a positive target to shoot.
Apply to upstream iDM source before compilation, requiring exact source anchors.
"""
from pathlib import Path
import sys

def main(root: Path):
    target=root/"source/input.c"
    source=target.read_text()
    old="""  case Omega_dcdmdr:
  case omega_dcdmdr:
  case Omega_scf:
  case Omega_ini_dcdm:
  case omega_ini_dcdm:
    /* Check that Omega's or omega's are nonzero: */
    if (target_value == 0.)
      *needs_shooting = _FALSE_;
    break;
  default:"""
    new="""  case Omega_dcdmdr:
  case omega_dcdmdr:
  case Omega_ini_dcdm:
  case omega_ini_dcdm:
    /* A zero physical target does not require inverse shooting. */
    if (target_value == 0.)
      *needs_shooting = _FALSE_;
    break;
  case Omega_scf:
    /* Omega_scf < 0 is the documented unknown-to-fill closure convention.
       Never shoot on an unknown physical scalar fraction. */
    if (target_value <= 0.)
      *needs_shooting = _FALSE_;
    break;
  default:"""
    if source.count(old)!=1:
        raise SystemExit("Refusing to patch: unexpected upstream source layout")
    target.write_text(source.replace(old,new))
    print("Patched negative Omega_scf closure: input shooting disabled for nonpositive targets",flush=True)

if __name__=="__main__":
    main(Path(sys.argv[1]))
