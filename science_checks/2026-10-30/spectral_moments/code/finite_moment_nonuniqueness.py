#!/usr/bin/env python3
"""Exact roots-of-unity obstruction to extracting spectral edge from n<=9."""
import numpy as np,json
from pathlib import Path
out=[]
for m in (10,11):
    phases=(2*np.arange(m)+0.5)*np.pi/m  # strictly between 0 and 2pi
    moment=np.array([np.mean(np.exp(-1j*n*phases)) for n in range(10)])
    assert np.max(np.abs(moment[1:]))<1e-13
    out.append({'atom_count':m,'strict_positive_minimum_energy':float(min(phases)),
                'max_abs_first_nine_moments':float(np.max(np.abs(moment[1:]))),
                'family_embeddable':'tensor with I_6 gives six-source equal moments up to n=9'})
p=Path(__file__).resolve().parents[1]/'results/finite_moment_no_go.json'
p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
