"""Fail-closed readiness check for a proposed QICT Planck/ACT posterior run."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

def report(theory_adapter, likelihood_root):
    adapter = Path(theory_adapter)
    data_root = Path(likelihood_root)
    required = {
        "qict_boltzmann_adapter": adapter.is_file(),
        "external_likelihood_data": data_root.is_dir() and any(data_root.iterdir()),
        "camb_or_class": any(importlib.util.find_spec(x) is not None for x in ("camb", "classy")),
        "cobaya": importlib.util.find_spec("cobaya") is not None,
        "act_dr6_lenslike": importlib.util.find_spec("act_dr6_lenslike") is not None,
    }
    return {"ready": all(required.values()), "checks": required,
            "status": "No full Planck/ACT likelihood evaluation or posterior is claimed."}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory-adapter", default="MISSING_QICT_SOLVER.py")
    ap.add_argument("--likelihood-root", default="MISSING_LIKELIHOOD_DATA")
    args = ap.parse_args()
    result = report(args.theory_adapter, args.likelihood_root)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ready"] else 2)
