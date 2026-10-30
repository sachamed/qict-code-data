#!/usr/bin/env python3
"""Check the SHA-256 manifest of canonical validation inputs against committed bytes."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "PROVENANCE.csv").open(encoding="utf-8", newline="")))
assert rows, "The provenance manifest is empty."
for row in rows:
    path = ROOT / row["artifact"]
    assert path.is_file(), f"Missing validation artifact: {row['artifact']}"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == row["source_sha256"], f"SHA-256 mismatch: {row['artifact']}"
print(f"PASS: {len(rows)} independently verified SHA-256 inputs")
