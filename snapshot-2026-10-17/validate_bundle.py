#!/usr/bin/env python3
"""Validate SHA-256 checksums for the scientific sources and data package."""
from pathlib import Path
import json,hashlib,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SHA256_MANIFEST.json').read_text())
bad=[]
for name,expected in manifest.items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:bad.append(name)
print(f'Verified {len(manifest)-len(bad)}/{len(manifest)} packaged scientific files.')
if bad:
 print('Checksum failures:',*bad,sep='\n')
 sys.exit(1)
