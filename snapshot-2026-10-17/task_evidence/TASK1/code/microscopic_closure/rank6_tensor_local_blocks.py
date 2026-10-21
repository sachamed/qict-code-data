#!/usr/bin/env python3
"""Exact local representation tensors for the QICT L=2 rank-six compiler.

This module does not perform the global 24-link contraction.  It provides the
objects that make that contraction deterministic: character weights, SU(2)
fusion/intertwiner channels, U(1) Gauss blocks, and symmetry-preserving SVD
truncation with an explicit discarded-norm certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np
from scipy.special import jv

PHI=math.pi/63.0
X1=4.0*PHI
X2=32.0*PHI


def u1_weight(k:int)->complex:
    # The common exp(-i x) zero-cost phase is kept outside the local character block.
    return (1j**k)*jv(k,X1)


def su2_weight(j2:int)->complex:
    """c_j for twice-spin j2=2j."""
    d=j2+1
    return (1j**j2)*(2.0*d/X2)*jv(d,X2)


def fuse(a2:int,b2:int):
    return range(abs(a2-b2),a2+b2+1,2)


def fusion_paths_to_zero(spins2:tuple[int,...]):
    """All left-associated SU(2) fusion paths ending in total spin zero.

    A path stores the intermediate twice-spins after each fusion.  SU(2) tensor
    products are multiplicity free; different paths are an orthogonal
    intertwiner basis after standard Clebsch-Gordan normalization.
    """
    if not spins2:
        return [()] 
    states={(spins2[0],): [()]}
    # simpler explicit recursion: list tuples (current_spin, path)
    current=[(spins2[0],())]
    for s in spins2[1:]:
        nxt=[]
        for a,path in current:
            for b in fuse(a,s):
                nxt.append((b,path+(b,)))
        current=nxt
    return [path for total,path in current if total==0]


def invariant_multiplicity(spins2:tuple[int,...])->int:
    return len(fusion_paths_to_zero(spins2))


@dataclass(frozen=True)
class LinkBlock:
    plaquette_spins2: tuple[int,int,int,int]
    matter_spin2: int
    u1_plaquette_flux: tuple[int,int,int,int]
    matter_charge: int
    matter_current: int

    def u1_allowed(self)->bool:
        # All four incident orientations have been absorbed into the signed fluxes.
        return sum(self.u1_plaquette_flux)+self.matter_charge*self.matter_current==0

    def su2_intertwiner_count(self)->int:
        spins=self.plaquette_spins2 + (() if self.matter_spin2==0 else (self.matter_spin2,))
        return invariant_multiplicity(spins)


def block_svd(matrix_blocks:dict[tuple, np.ndarray], rel_tol:float=0.0, abs_budget:float=1e-8):
    """Symmetry-preserving SVD of independent charge/spin blocks.

    Singular values are never mixed across sectors. The sum of discarded
    singular values must stay within abs_budget. A positive rel_tol additionally
    protects values above that fraction of their own block's largest value;
    rel_tol=0 uses the absolute budget alone.

    The discarded sum bounds the truncation error in exact arithmetic.
    Floating-point SVD and reconstruction errors require separate validation
    before this diagnostic can be used as a physical error certificate.
    """
    if not math.isfinite(rel_tol) or rel_tol < 0:
        raise ValueError("rel_tol must be finite and nonnegative")
    if not math.isfinite(abs_budget) or abs_budget < 0:
        raise ValueError("abs_budget must be finite and nonnegative")
    svds={}
    spectra=[]
    for q,A in matrix_blocks.items():
        U,s,Vh=np.linalg.svd(A,full_matrices=False)
        svds[q]=(U,s,Vh)
        for i,val in enumerate(s): spectra.append((float(val),q,i))
    # Discard from smallest upward while respecting budget. Spectral norm of a
    # block truncation is the largest discarded singular value in that block;
    # using the sum of all discarded singular values is conservative.
    spectra.sort(key=lambda x:x[0])
    discarded=set(); total=0.0
    for val,q,i in spectra:
        if rel_tol > 0 and val > rel_tol*svds[q][1][0]:
            continue
        if total+val > abs_budget:
            # Later candidates are no smaller, so none can fit the budget.
            break
        discarded.add((q,i)); total+=val
    out={}
    for q,(U,s,Vh) in svds.items():
        keep=[i for i in range(len(s)) if (q,i) not in discarded]
        out[q]=(U[:,keep],s[keep],Vh[keep,:])
    return out, {'discarded_singular_value_sum_bound':total,'absolute_budget':abs_budget}


def self_test():
    # Character coefficients are nonzero and fusion multiplicities reproduce
    # elementary SU(2) facts.
    assert abs(u1_weight(0))>0
    assert abs(su2_weight(0))>0
    assert invariant_multiplicity((1,1))==1  # 1/2 x 1/2 -> 0
    assert invariant_multiplicity((1,1,1))==0
    assert invariant_multiplicity((1,1,1,1))==2
    # A left fundamental matter line requires the four gauge spins to have odd
    # total twice-spin parity. Example: 0,0,0,1 plus matter 1 -> singlet.
    assert invariant_multiplicity((0,0,0,1,1))==1
    # U(1) Gauss block sanity check.
    b=LinkBlock((0,0,0,1),1,(2,-1,5,0),-6,1)
    assert b.u1_allowed()  # 2-1+5-6=0
    assert b.su2_intertwiner_count()==1
    # Block SVD sanity.
    A={('q0','j0'):np.diag([1.0,1e-9]),('q1','j1'):np.diag([0.5,2e-9])}
    _,cert=block_svd(A,abs_budget=1e-8)
    assert cert['discarded_singular_value_sum_bound']<=1e-8
    # A relative threshold must never override the global absolute budget.
    # The previous OR condition discarded 0.11 and 0.08, giving operator
    # error 0.11 despite an absolute budget of 0.10.
    A={('relative_budget',):np.diag([1.0,0.11,0.08])}
    factors,cert=block_svd(A,rel_tol=0.2,abs_budget=0.1)
    U,s,Vh=factors[('relative_budget',)]
    assert len(s)==2
    assert cert['discarded_singular_value_sum_bound']<=0.1
    assert np.linalg.norm(A[('relative_budget',)]-(U*s)@Vh,2)<=0.1
    # Relative protection is evaluated within each charge block.
    A={('small_block',):np.diag([0.001]),('large_block',):np.diag([100.0])}
    factors,cert=block_svd(A,rel_tol=0.01,abs_budget=1.0)
    assert all(len(parts[1])==1 for parts in factors.values())
    assert cert['discarded_singular_value_sum_bound']==0.0
    for kwargs in ({'rel_tol':-1.0},{'abs_budget':-1.0},
                   {'rel_tol':float('nan')},{'abs_budget':float('inf')}):
        try:
            block_svd({},**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid SVD truncation tolerance accepted")
    return {
      'status':'PASS_LOCAL_BLOCKS',
      'tests':10,
      'u1_x':X1,'su2_x':X2,
      'example_four_fundamental_intertwiner_multiplicity':invariant_multiplicity((1,1,1,1))
    }

if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
