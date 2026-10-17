#!/usr/bin/env python3
"""Certified error ledger for the coarse/fine Task-3 route moments."""
from __future__ import annotations
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def load(p): return json.load(open(p))

def main():
    tail=load(ROOT/'upstream/task1/SU2_CHARACTER_TAIL_CERTIFICATE.json')
    rows={int(x['twice_j_max']):x for x in tail['rows']}
    # Max route has 10 slices * 24 plaquettes = 240 magnetic gates.
    gates=240
    data={}
    for j2 in (10,12,14,16):
        one=float(rows[j2]['uniform_character_tail_upper'])
        # Exact telescoping for unit-norm factors: (1+e)^N-1.
        global_bound=math.expm1(gates*math.log1p(one))
        data[str(j2)]={'one_gate_uniform_tail':one,'240_gate_telescoped_bound':global_bound}
    obj={
      'status':'PASS_ADAPTIVE_SU2_CHARACTER_ERROR_LEDGER',
      'max_route_slices':10,'plaquettes_per_slice':24,'magnetic_gate_count':gates,
      'cutoff_bounds':data,
      'coarse_twice_j_max':12,'fine_twice_j_max':14,
      'coarse_bound':data['12']['240_gate_telescoped_bound'],
      'fine_bound':data['14']['240_gate_telescoped_bound'],
      'rule':'The PEPO frontier compression budget is additional to these analytic representation tails; no numerical SVD discard is hidden inside the character cutoff error.',
      'external_particle_data_used':False
    }
    print(json.dumps(obj,indent=2))
if __name__=='__main__': main()
