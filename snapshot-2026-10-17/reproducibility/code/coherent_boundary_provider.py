#!/usr/bin/env python3
"""Coherent boundary contraction without treating intertwiners as charges.

This provides exact finite input/output interfaces and a test on a genuine
archived SU(2) recoupling column. It does not synthesize the missing L24 state
or identify that local recoupling with the complete charged C10 circuit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/coherent_boundary'


@dataclass(frozen=True)
class ChargeSector:
    qY: int
    twice_j: int
    su3_p: int
    su3_q: int
    CAR_parity: int

    def validate(self):
        assert self.twice_j>=0 and self.su3_p>=0 and self.su3_q>=0
        assert self.CAR_parity in (0,1)


@dataclass(frozen=True)
class BoundaryBasis:
    # Every row has a charge sector and a coherent multiplicity coordinate.
    # iota2/iota3 are row coordinates inside a charge block, never implicit
    # classical measurement outcomes.
    charges: tuple[ChargeSector,...]
    multiplicity_labels: tuple[tuple[int,...],...]
    provenance: str

    def validate(self):
        assert len(self.charges)==len(self.multiplicity_labels)>0
        for charge in self.charges:charge.validate()
        keys=[(c,m) for c,m in zip(self.charges,self.multiplicity_labels)]
        assert len(set(keys))==len(keys)

    @property
    def dimension(self):return len(self.charges)


@dataclass(frozen=True)
class CoherentDensity:
    basis: BoundaryBasis
    gram_factor: np.ndarray

    def validate(self,tolerance=1e-11):
        self.basis.validate()
        factor=np.asarray(self.gram_factor,complex)
        assert factor.ndim==2 and factor.shape[0]==self.basis.dimension
        assert np.isfinite(factor).all()
        assert abs(np.vdot(factor,factor).real-1)<tolerance

    def matrix(self):
        self.validate()
        return self.gram_factor@self.gram_factor.conj().T


@dataclass(frozen=True)
class StinespringShell:
    input_basis: BoundaryBasis
    output_basis: BoundaryBasis
    # One *common* environment index: V[out,environment,in].
    isometry: np.ndarray
    radius_in: int
    radius_out: int
    provenance: str
    certified_isometry_norm_error: float=0.

    def validate(self,tolerance=1e-11):
        self.input_basis.validate();self.output_basis.validate()
        assert self.radius_out<self.radius_in
        v=np.asarray(self.isometry,complex)
        assert v.ndim==3 and v.shape[0]==self.output_basis.dimension
        assert v.shape[2]==self.input_basis.dimension and v.shape[1]>0
        flat=v.reshape(-1,v.shape[-1])
        error=float(np.linalg.norm(flat.conj().T@flat-np.eye(v.shape[-1]),2))
        assert error<=tolerance+self.certified_isometry_norm_error,error
        return error


def contract_shell(density:CoherentDensity,shell:StinespringShell):
    density.validate();error=shell.validate()
    assert density.basis==shell.input_basis
    # Low-rank propagation: rho_out=sum_e (K_e L)(K_e L)^dagger.
    # Concatenating columns retains every off-diagonal multiplicity entry.
    factors=[shell.isometry[:,e,:]@density.gram_factor
             for e in range(shell.isometry.shape[1])]
    out=CoherentDensity(shell.output_basis,np.concatenate(factors,axis=1))
    out.validate(tolerance=1e-11+shell.certified_isometry_norm_error)
    return out,{'radius_in':shell.radius_in,'radius_out':shell.radius_out,
                'isometry_defect_diagnostic':error,
                'environment_dimension':shell.isometry.shape[1],
                'output_Gram_rank_columns':out.gram_factor.shape[1],
                'trace':float(np.vdot(out.gram_factor,out.gram_factor).real),
                'hidden_dephasing_or_renormalization':False,
                'provenance':shell.provenance}


def contract_boundary_paths(density,effects):
    """Return Tr(F_p rho), including coherent off-diagonal path effects."""
    density.validate()
    factor=density.gram_factor
    values=[]
    for effect in effects:
        a=sparse.csr_matrix(effect,dtype=complex)
        assert a.shape==(density.basis.dimension,)*2
        values.append(np.vdot(factor,a@factor))
    return np.asarray(values)


def four_spin_basis():
    # Computational bit convention: 0=up,1=down. Both vectors are total-spin
    # zero. The second couples (12),(34) through spin one to a singlet.
    singlet=np.zeros(16,complex);triplet=np.zeros(16,complex)
    for bits,coefficient in [('0101',.5),('0110',-.5),('1001',-.5),('1010',.5)]:
        singlet[int(bits,2)]=coefficient
    for bits in ['0011','1100']:triplet[int(bits,2)]=1/np.sqrt(3)
    for bits in ['0101','0110','1001','1010']:triplet[int(bits,2)]=-1/(2*np.sqrt(3))
    input_singlet=np.zeros(16,complex)
    for bits,coefficient in [('0011',.5),('0101',-.5),('1010',-.5),('1100',.5)]:
        input_singlet[int(bits,2)]=coefficient
    basis=np.stack([triplet,singlet],axis=1)
    return basis,input_singlet


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    payload=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    with np.load(payload,allow_pickle=False) as z:
        archived=z['su2_connected_branch_amplitudes'][0]
        branch_codes=z['su2_connected_branch_code'][0].tolist()
    basis,psi=four_spin_basis()
    column=basis.conj().T@psi
    assert np.linalg.norm(column-archived)<3e-15
    assert np.linalg.norm(basis.conj().T@basis-np.eye(2))<2e-15
    # Full change of invariant basis: second input column is the normalized
    # orthogonal singlet, with its phase fixed in this recorded convention.
    recoupler=np.array([[np.sqrt(3)/2,-.5],[-.5,-np.sqrt(3)/2]],complex)
    assert np.linalg.norm(recoupler[:,0]-column)<2e-15
    sector=ChargeSector(0,0,0,0,0)
    carrier=BoundaryBasis((sector,sector),((2,0),(0,0)),
                          'same total SU2 singlet; two fusion multiplicities')
    source=CoherentDensity(carrier,np.array([[1.],[0.]],complex))
    shell=StinespringShell(carrier,carrier,recoupler[:,None,:],2,1,
                            'actual four-fundamental SU2 Haar intertwiner basis change')
    coherent,check=contract_shell(source,shell)
    rho=coherent.matrix()
    dephased=np.diag(np.diag(rho))
    returned=recoupler.conj().T@rho@recoupler
    legacy_return=recoupler.conj().T@dephased@recoupler
    assert abs(returned[0,0]-1)<2e-15
    assert abs(legacy_return[0,0]-5/8)<2e-15
    distance=.5*np.linalg.norm(rho-dephased,ord='nuc')
    assert abs(distance-np.sqrt(3)/4)<2e-15
    # The off-diagonal effect directly tests retention of the coherent cross
    # term; diagonal branch probabilities cannot evaluate this observable.
    effect=np.array([[0,1],[1,0]],complex)
    weights=contract_boundary_paths(coherent,[effect,np.eye(2)])
    assert abs(weights[0]+np.sqrt(3)/2)<2e-15
    # Gauge invariance is checked on actual spin tensors, independently of
    # the two-dimensional recoupling matrices.
    rng=np.random.default_rng(63);gauge_error=0.
    for _ in range(24):
        x=rng.normal(size=4);x/=np.linalg.norm(x)
        u=np.array([[x[0]+1j*x[3],x[2]+1j*x[1]],[-x[2]+1j*x[1],x[0]-1j*x[3]]])
        four=u
        for k in range(3):four=np.kron(four,u)
        gauge_error=max(gauge_error,float(np.linalg.norm(four@basis-basis)))
    assert gauge_error<4e-15
    # A nonisometric shell is rejected before it can claim trace preservation.
    rejected=False
    try:
        bad=StinespringShell(carrier,carrier,.9*recoupler[:,None,:],2,1,'invalid contraction test')
        bad.validate()
    except AssertionError:rejected=True
    assert rejected
    npz=OUT/'COHERENT_SU2_BOUNDARY_INTERTWINER_DATA.npz'
    np.savez_compressed(npz,spin_tensor_basis=basis,input_pair_singlet=psi,
                        recoupler=recoupler,coherent_density=rho,
                        dephased_density=dephased,coherent_return=returned,
                        dephased_return=legacy_return,path_effects=weights,
                        archived_recoupling_column=archived)
    result={'status':'COHERENT_STINESPRING_BOUNDARY_INTERFACE_VERIFIED__L24_LOCAL_INTERTWINER_TEST',
            'payload_sha256':hashlib.sha256(payload.read_bytes()).hexdigest(),
            'archived_pair_index':0,'branch_codes_twice_j':branch_codes,
            'charge_sector':['qY=0','total_j=0','SU3=(0,0)','CAR_parity=0'],
            'coherent_multiplicity_coordinates':['j12=1','j12=0'],
            'pure_recoupled_column_exact':['sqrt(3)/2','-1/2'],
            'coherent_density_off_diagonal_exact':'-sqrt(3)/4',
            'coherent_round_trip_return_probability_exact':'1',
            'iota_dephased_round_trip_return_probability_exact':'5/8',
            'iota_dephasing_trace_distance_exact':'sqrt(3)/4',
            'archived_column_error':float(np.linalg.norm(column-archived)),
            'four_spin_gauge_invariance_error':gauge_error,
            'coherent_shell_checks':check,'nonisometric_shell_rejected':rejected,
            'full_quantum_C10_shell_map_derived':False,
            'physical_scope':'An actual archived local SU2 intertwiner and a complete '
                             'finite coherent-density/Kraus input interface. The basis-change '
                             'test is not a C10 quantum state or the full charged evolution.',
            'legacy_interface_condition':'The older iota-labelled SectorDensity is valid '
                                         'only when those keys are independently justified '
                                         'superselection labels, or all coherent iota coordinates '
                                         'are packed inside one block. Gauge symmetry alone '
                                         'does not superselect intertwiner multiplicities.',
            'required_global_input':'A coherent selected boundary factor and Stinespring '
                                    'tensors obtained from the full ordered matter-gauge circuit, '
                                    'including SU3 labels and the common environment indices.',
            'data_sha256':hashlib.sha256(npz.read_bytes()).hexdigest()}
    (OUT/'COHERENT_BOUNDARY_CERTIFICATE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'coherent_return':returned[0,0].real,'dephased_return':legacy_return[0,0].real,
                      'dephasing_trace_distance':distance,'gauge_invariance_error':gauge_error,
                      'full_C10':False},indent=2))


if __name__=='__main__':main()
