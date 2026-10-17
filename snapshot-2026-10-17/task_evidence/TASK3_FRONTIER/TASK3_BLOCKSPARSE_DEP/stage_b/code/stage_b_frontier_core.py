#!/usr/bin/env python3
"""Symmetry-block boundary compression core for QICT-D21 Task 3.

This module is deliberately representation-agnostic: it implements the part of
Stage B that can be tested without inventing missing physical tensors.  Physical
blocks must be supplied by the rank-six PEPO compiler/recoupler.  The code never
mixes U(1), SU(2), intertwiner or CAR sectors and returns an explicit 2-norm
truncation certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import json
import numpy as np

Sector = Tuple[int,int,int,int]  # (q_U1, twice_j_SU2, intertwiner, CAR parity)

@dataclass
class TruncationCertificate:
    discarded_frobenius_sq: float
    discarded_two_norm_bound: float
    kept_rank: int
    original_rank: int
    sectors: int


def _svd_rank_for_budget(s: np.ndarray, budget_sq: float, rel_tol: float) -> int:
    if s.size == 0:
        return 0
    mandatory = int(np.sum(s > rel_tol * s[0])) if rel_tol > 0 else len(s)
    # If relative tolerance is zero, budget controls. Otherwise never discard a
    # singular value above relative threshold.
    rmin = mandatory if rel_tol > 0 else 0
    # Choose smallest retained rank whose Frobenius tail is inside budget.
    tail = np.cumsum((s[::-1]**2))[::-1]
    for r in range(rmin, len(s)+1):
        t = 0.0 if r == len(s) else float(tail[r])
        if t <= budget_sq:
            return r
    return len(s)


def compress_blocks(blocks: Dict[Sector,np.ndarray], abs_two_norm_budget: float,
                    rel_tol: float = 0.0):
    """SVD each exact symmetry sector independently.

    The global error certificate uses ||direct_sum Delta_s||_2 <=
    ||direct_sum Delta_s||_F = sqrt(sum_s ||Delta_s||_F^2).  Budget is allocated
    adaptively by globally sorting singular-value squares, but symmetry labels
    are never mixed.
    """
    if abs_two_norm_budget < 0:
        raise ValueError('budget must be nonnegative')
    decomp = {}
    all_sv = []
    original_rank = 0
    for sec,A in blocks.items():
        A=np.asarray(A,dtype=np.complex128)
        if A.ndim != 2:
            raise ValueError(f'block {sec} must be a matrix')
        U,s,Vh=np.linalg.svd(A,full_matrices=False)
        decomp[sec]=(U,s,Vh)
        original_rank += len(s)
        for i,x in enumerate(s):
            # singular values protected by relative threshold are not eligible
            protected = bool(rel_tol>0 and x > rel_tol*s[0])
            all_sv.append((float(x*x),sec,i,protected))
    # Discard globally smallest eligible singular values while total Frobenius
    # tail remains within budget^2. This is conservative for operator 2-norm.
    all_sv.sort(key=lambda t:t[0])
    discard=set(); used=0.0; B2=abs_two_norm_budget**2
    for sq,sec,i,protected in all_sv:
        if protected:
            continue
        if used + sq <= B2*(1+1e-15):
            discard.add((sec,i)); used += sq
        else:
            break
    factors={}; reconstructed={}; kept=0
    for sec,(U,s,Vh) in decomp.items():
        idx=[i for i in range(len(s)) if (sec,i) not in discard]
        kept += len(idx)
        factors[sec]=(U[:,idx],s[idx],Vh[idx,:])
        reconstructed[sec]=(U[:,idx]*s[idx])@Vh[idx,:] if idx else np.zeros((U.shape[0],Vh.shape[1]),complex)
    cert=TruncationCertificate(
        discarded_frobenius_sq=float(used),
        discarded_two_norm_bound=float(np.sqrt(used)),
        kept_rank=kept, original_rank=original_rank, sectors=len(blocks))
    return factors,reconstructed,cert


def compose_error_bounds(step_bounds):
    """Triangle-inequality accumulator for normalized boundary sweeps."""
    return float(sum(float(x) for x in step_bounds))


def self_test():
    rng=np.random.default_rng(210921)
    blocks={
      (0,0,0,0):np.diag([1.0,0.25,1e-9]),
      (-3,1,0,1):np.diag([0.7,0.1,2e-9]),
      (-6,0,0,1):np.diag([0.6,0.05,3e-9]),
    }
    budget=1e-8
    fac,rec,cert=compress_blocks(blocks,budget,rel_tol=0.0)
    # exact sector preservation
    assert set(fac)==set(blocks)
    # direct numerical Frobenius error is bounded by certificate and budget
    err2=sum(float(np.linalg.norm(blocks[s]-rec[s],'fro')**2) for s in blocks)
    assert abs(err2-cert.discarded_frobenius_sq) < 1e-24
    assert cert.discarded_two_norm_bound <= budget*(1+1e-12)
    # no symmetry block is ever concatenated/mixed
    assert fac[(0,0,0,0)][0].shape[0] == 3
    # deterministic random block regression
    A=rng.normal(size=(7,5))+1j*rng.normal(size=(7,5))
    _,r2,c2=compress_blocks({(2,2,1,0):A},0.0)
    assert np.linalg.norm(A-r2[(2,2,1,0)]) < 1e-12
    return {
      'status':'PASS_STAGEB_BLOCK_SPARSE_FRONTIER_CORE',
      'sector_label':['q_U1','twice_j_SU2','iota_SU2','Pi_CAR'],
      'tests':5,
      'test_budget':budget,
      'certified_discarded_two_norm_bound':cert.discarded_two_norm_bound,
      'zero_budget_reconstruction_error':float(np.linalg.norm(A-r2[(2,2,1,0)])),
      'scientific_scope':'Compression/error-accounting core only; physical SU(2) 24-link recoupling tensors are not synthesized here.'
    }

if __name__=='__main__':
    print(json.dumps(self_test(),indent=2))
