#!/usr/bin/env python3
"""Independent finite tests for the moment-to-threshold inequalities."""
from pathlib import Path
import json,math
import numpy as np
from scipy.stats import wasserstein_distance
R=Path(__file__).resolve().parents[1]/'results'
checks={};details=[]
def check(name,x):checks[name]=bool(x)
rng=np.random.default_rng(1729)
M=2047;N=M+1;k=np.arange(1,N)
for j in range(30):
    # Atom at the lower edge with a known, fixed positive weight.
    a=rng.uniform(0,.2);shift=rng.uniform(1e-4,.04)
    x=np.r_[a,a+rng.uniform(.1,.6,5)]
    p=np.r_[.4,rng.dirichlet(np.ones(5))*.6];y=x+shift
    c=np.exp(1j*np.outer(k,x))@p;d=np.exp(1j*np.outer(k,y))@p
    budget=2*math.pi*(.5+math.log(N))/N+2*np.sum((1-k/N)*np.abs(c-d)/k)
    w=wasserstein_distance(x,y,p,p)
    check(f'atomic_moment_bound_{j}',w<=budget+1e-14)
    check(f'atomic_edge_transport_bound_{j}',shift<=2*w/.4+1e-14)
    details.append({'kind':'shifted_atomic','W1':float(w),'moment_bound':float(budget),'edge_difference':float(shift),'visibility_c':.4,'visibility_kappa':0})
# Continuous uniform measures of width L: visibility c=1/L, kappa=1.
L=.8;shift=.01;a=.1
c=np.exp(1j*k*(a+L/2))*np.sinc(k*L/(2*np.pi));d=c*np.exp(1j*k*shift)
budget=2*math.pi*(.5+math.log(N))/N+2*np.sum((1-k/N)*np.abs(c-d)/k)
check('continuous_moment_bound',shift<=budget)
check('continuous_small_error_domain',budget<(1/L)*L**2)
check('continuous_edge_bound',shift<=2*math.sqrt(budget*L))
details.append({'kind':'shifted_uniform','W1':shift,'moment_bound':float(budget),'edge_difference':shift,'visibility_c':1/L,'visibility_kappa':1})
# The tail weight tends to zero, so no uniform visibility constant exists.
for eps in [1e-2,1e-5,1e-8]:
    c=(1-eps)*np.exp(1j*k)+eps;d=np.exp(1j*k)
    check('vanishing_tail_'+str(eps),float(np.max(abs(c-d)))<=2*eps+1e-15 and eps>0)
# Verify an operator-norm error controls normalized trace error for noncommuting matrices.
for j in range(20):
    E=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
    check(f'trace_scalarization_{j}',abs(np.trace(E))/6<=np.linalg.norm(E,2)+1e-13)
out={'classification':'VISIBLE_THRESHOLD_BOUND_TESTS','pass':all(checks.values()),'check_count':len(checks),'checks':checks,'examples':details,'scope':'Finite independent tests of proved estimates; no empirical mass prediction or continuum certificate.'}
(R/'VISIBLE_THRESHOLD_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'pass':out['pass'],'check_count':len(checks)}))
if not out['pass']:raise SystemExit(2)
