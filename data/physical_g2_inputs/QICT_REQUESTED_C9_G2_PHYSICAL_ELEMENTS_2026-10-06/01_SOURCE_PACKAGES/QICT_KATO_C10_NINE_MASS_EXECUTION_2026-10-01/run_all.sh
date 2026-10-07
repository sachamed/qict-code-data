#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 code/validate_kato_c10.py > results/PREFLIGHT_RUN.log
cat results/PREFLIGHT_STATUS.json >> results/PREFLIGHT_RUN.log
# validate_kato_c10.py prints the same JSON it writes; retain one canonical copy in the log.
python3 - <<'PY2'
from pathlib import Path
p=Path("results/PREFLIGHT_STATUS.json")
log=Path("results/PREFLIGHT_RUN.log")
log.write_text(p.read_text())
PY2

if [[ -f input/G2_NINE_FLAVOR_PAYLOAD.json ]]; then
  python3 code/extract_nine_masses.py | tee -a results/PREFLIGHT_RUN.log
else
  cat >> results/PREFLIGHT_RUN.log <<'EOF'

G2 physical payload not yet present.
Copy input/G2_NINE_FLAVOR_PAYLOAD.template.json to input/G2_NINE_FLAVOR_PAYLOAD.json
only after physical Kato/C10 G2 values on L=4,8,12 have been computed.
No masses were fabricated.
EOF
fi
