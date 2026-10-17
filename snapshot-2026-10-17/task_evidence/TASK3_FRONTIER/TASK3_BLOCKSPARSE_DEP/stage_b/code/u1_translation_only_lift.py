#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'results'
z=np.load(DATA/'U1_L2_KATO_S24_STATE.npz')
reps=np.asarray(z['reps'],dtype=np.int16)
sizes_full=np.asarray(z['sizes'],dtype=np.int16)
v=np.asarray(z['v'],dtype=np.complex128)
L=2
links=[(mu,x,y,zz) for mu in range(3) for x in range(L) for y in range(L) for zz in range(L)]
li={t:i for i,t in enumerate(links)}
TRANS=list(itertools.product(range(2),repeat=3))
PERMS=list(itertools.permutations(range(3)))

def map_inv_translation(t):
    no=np.empty(24,dtype=np.int64)
    for old,(mu,x,y,zz) in enumerate(links):
        c=((x+t[0])%2,(y+t[1])%2,(zz+t[2])%2)
        no[old]=li[(mu,*c)]
    inv=np.empty(24,dtype=np.int64); inv[no]=np.arange(24)
    return inv
TINV=[map_inv_translation(t) for t in TRANS]

def permute(E,perm):
    no=np.empty(24,dtype=np.int64)
    invp=[perm.index(i) for i in range(3)]
    for old,(mu,x,y,zz) in enumerate(links):
        c=(x,y,zz); cp=[c[invp[i]] for i in range(3)]
        no[old]=li[(perm[mu],*cp)]
    inv=np.empty(24,dtype=np.int64); inv[no]=np.arange(24)
    return E[inv]

def tcanon(E):
    cand=[tuple(map(int,E[inv])) for inv in TINV]
    return min(cand)

def tsize(rep):
    return len({tuple(map(int,rep[inv])) for inv in TINV})

# Every full translation x S3 orbit splits into at most six translation orbits.
treps=[]; owner=[]
for i,E in enumerate(reps):
    loc={tcanon(permute(E,p)) for p in PERMS}
    for r in sorted(loc):
        treps.append(r); owner.append(i)
# A full-orbit representative can share a translation suborbit with no other full orbit.
# Deduplicate defensively and check ownership consistency.
seen={}; conflicts=0
for r,o in zip(treps,owner):
    if r in seen and seen[r]!=o: conflicts+=1
    else: seen[r]=o
assert conflicts==0
tr=sorted(seen)
own=np.array([seen[r] for r in tr],dtype=np.int32)
ts=np.array([tsize(np.asarray(r,dtype=np.int16)) for r in tr],dtype=np.int16)
# The full orbit is a disjoint union of its translation suborbits.
parts=np.zeros(len(reps),dtype=np.int32)
weightsum=np.zeros(len(reps),dtype=np.int32)
for o,s in zip(own,ts): parts[o]+=1; weightsum[o]+=int(s)
assert np.array_equal(weightsum.astype(np.int16),sizes_full)
# Lift a cubic-symmetric normalized orbit vector to the translation-orbit basis.
vt=v[own]*np.sqrt(ts.astype(float)/sizes_full[own].astype(float))
norm0=float(np.linalg.norm(v)); normt=float(np.linalg.norm(vt))
# Reconstruct the original coefficient by projection back onto the normalized full orbit.
vback=np.zeros_like(v)
for a,o in enumerate(own):
    vback[o]+=vt[a]*math.sqrt(float(ts[a])/float(sizes_full[o]))
backerr=float(np.linalg.norm(vback-v))
# Counts by splitting multiplicity and orbit size.
sp={str(k):int(np.sum(parts==k)) for k in sorted(set(map(int,parts)))}
osz={str(k):int(np.sum(ts==k)) for k in sorted(set(map(int,ts)))}
out={
 'status':'PASS_U1_TRANSLATION_ONLY_SYMMETRY_LIFT',
 'input_full_symmetry_orbit_dim':int(len(reps)),
 'translation_only_orbit_dim':int(len(tr)),
 'dimension_increase_factor':float(len(tr)/len(reps)),
 'full_flux_basis_dim':int(np.sum(sizes_full)),
 'translation_orbit_size_histogram':osz,
 'full_orbit_translation_suborbit_count_histogram':sp,
 'max_suborbits_per_full_orbit':int(parts.max()),
 'orbit_partition_exact':bool(np.array_equal(weightsum.astype(np.int16),sizes_full)),
 'input_vector_norm':norm0,
 'lifted_vector_norm':normt,
 'norm_residual':abs(normt-norm0),
 'roundtrip_projection_residual':backerr,
 'scientific_scope':'Exact symmetry lift of the existing S=24 U(1) Kato regression from translations x cubic permutations to translations only. This removes the illegal axis-permutation quotient for ordered S_z S_y S_x diagnostics, but S=24 remains an auxiliary Galerkin regression and is not the final PEPO physical regulator.'
}
# Save compact arrays for subsequent branch-conditioned U1 diagnostics.
np.savez_compressed(DATA/'U1_L2_KATO_S24_TRANSLATION_ONLY.npz', reps=np.asarray(tr,dtype=np.int16), sizes=ts, owner=own, v=vt)
(OUT/'U1_TRANSLATION_ONLY_LIFT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
