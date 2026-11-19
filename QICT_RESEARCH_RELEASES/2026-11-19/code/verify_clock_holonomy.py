#!/usr/bin/env python3
"""Cycle-holonomy phase reconstruction on a fixed finite geometry graph.
This is a graph-theoretic comparator, NOT a measured QICT seventh-cell amplitude.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
rng=np.random.default_rng(20261118)
edges=[(0,1),(1,2),(2,3),(3,0),(0,2)]
B=np.zeros((4,len(edges)))
for j,(a,b) in enumerate(edges):B[a,j]=-1;B[b,j]=1
Z=null_space(B)
assert np.linalg.matrix_rank(B)==3 and Z.shape==(5,2)
errors=[]
for _ in range(200):
    theta=rng.uniform(-.3,.3,len(edges))
    alpha=rng.uniform(-.3,.3,4)
    gauge=theta+B.T@alpha
    errors.append(np.linalg.norm(Z.T@theta-Z.T@gauge))
    # If 2 edge assignments yield identical loop holonomies they differ by an incidence gradient.
    diff=gauge-theta
    proj=B.T@np.linalg.lstsq(B.T,diff,rcond=None)[0]
    errors.append(np.linalg.norm(diff-proj))
assert max(errors)<1e-12
phase=np.array([.1,.2,.3,.4,.5]); delta=np.zeros(5);delta[4]=.3
shift=float(np.linalg.norm(Z.T@(phase+delta)-Z.T@phase))
assert shift>.01
out={'classification':'mathematical graph comparator, not actual microscopic seventh-cell holonomy',
     'vertices':4,'edges':len(edges),'cycle_rank':Z.shape[1],
     'random_comparisons':200,'largest_gauge_invariance_or_gradient_fit_error':max(errors),
     'nontrivial_cycle_phase_shift':shift,
     'theorem':'On a connected finite graph, full cycle holonomies determine edge U(1) phases up to vertex rephasing; specifying a clock reference fixes that last gauge.'}
p=Path(__file__).resolve().parent/'clock_holonomy_checks.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
