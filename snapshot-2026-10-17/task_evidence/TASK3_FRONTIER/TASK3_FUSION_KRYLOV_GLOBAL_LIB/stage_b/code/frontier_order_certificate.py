#!/usr/bin/env python3
from __future__ import annotations
import json
from su2_global_spin_network import PLAQ,LINKS
N=24
CANON=list(range(N))
ORDER=[16,18,20,22,14,10,11,15,5,4,6,7,13,9,1,3,21,17,0,2,19,23,8,12]

def profile(order):
    pos={e:i for i,e in enumerate(order)}; out=[]
    for c in range(N-1):
        n=0
        for p in PLAQ:
            q=[pos[e] for e in p['edges']]
            if min(q)<=c<max(q):n+=1
        out.append(n)
    return out

def metrics(order):
    p=profile(order)
    return {'max_open_plaquettes':max(p),'sum_open_plaquettes':sum(p),'sum_squares':sum(x*x for x in p),'profile':p}

def main():
    a=metrics(CANON);b=metrics(ORDER)
    assert sorted(ORDER)==list(range(24))
    assert b['max_open_plaquettes']<a['max_open_plaquettes']
    obj={'status':'PASS_FRONTIER_ORDER_WIDTH_REDUCTION','canonical':a,'optimized':b,
         'optimized_link_order':ORDER,'optimized_link_labels':[list(LINKS[e]) for e in ORDER],
         'max_width_reduction_fraction':1-b['max_open_plaquettes']/a['max_open_plaquettes'],
         'scope':'Deterministic certified quality of this ordering; no claim of global pathwidth optimality.'}
    print(json.dumps(obj,indent=2))
if __name__=='__main__':main()
