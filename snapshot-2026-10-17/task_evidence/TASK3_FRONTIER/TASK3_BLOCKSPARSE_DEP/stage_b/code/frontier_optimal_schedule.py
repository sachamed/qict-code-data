#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json
sys.path.insert(0,os.path.dirname(__file__))
import su2_global_spin_network as gs
ORDER=[0,1,2,3,8,9,12,13,16,18,20,22,4,5,6,7,10,11,14,15,17,19,21,23]

def main():
    pm=[set(p['edges']) for p in gs.PLAQ]; processed=set(); prev=set(); rows=[]; prof=[]
    for step,e in enumerate(ORDER):
        processed.add(e)
        op={p for p,E in enumerate(pm) if E&processed and not E<=processed}
        opened=sorted(op-prev); closed=sorted(prev-op); prof.append(len(op) if step<23 else 0)
        rows.append({'step':step+1,'link_id':e,'link':list(gs.LINKS[e]),'plaquettes_opened':opened,'plaquettes_closed':closed,'open_after':len(op),'open_set':sorted(op)})
        prev=op
    p23=[r['open_after'] for r in rows[:-1]]
    assert max(p23)==14 and not prev
    out={'status':'PASS_EXACT_WIDTH14_FRONTIER_SCHEDULE','order':ORDER,'max_open_before_final':max(p23),'profile_before_final':p23,
         'steps':rows,
         'execution_rule':'At each link, fuse all incident already-open plaquette representation legs with the vertical electric/matter representation leg, resolve the SU(2) invariant fusion channel locally, close any plaquette whose fourth edge is reached, then block-SVD only across the surviving open-set sectors.',
         'symmetry_labels':['U1 electric charge','SU2 twice-j','SU2 fusion/intertwiner path','CAR parity','C8 internal branch/spin auxiliary'],
         'matter_reduction':'Use C8_L2_INTERNAL_REDUCTION: position is deterministic and only the 4-dimensional spin x weak block is propagated; only 12 physical links are matter-active over the two macro parities.'}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
