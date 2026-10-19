#!/usr/bin/env python3
"""Compare the unwrapped primitive H ground state with its *split-step* gates.

The Hamiltonian H=K_E+K_B has a certified isolated local pole, but its ground
state need not be invariant under exp(-i phi K_B) exp(-i phi K_E). This code
measures that logical distinction in the source-backed one-plaquette U(1)
sector. This is not the full interacting QICT U_micro.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.special import jv
from scipy.linalg import expm

from primitive_infinite_volume_certificate import ROOT,EVIDENCE,run as ground_run

PHI=np.pi/63.


def exact_infinite_link_matrix_on_source(flux):
    r=np.asarray(flux,int)
    delta=r[:,None]-r[None,:]
    # K_B=4-2(S+S*) on infinite flux space.
    # exp(-i phi K_B)[r,s] = exp(-i4phi) i^(r-s) J_(r-s)(4phi).
    return np.exp(-4j*PHI)*np.power(1j,delta)*jv(delta,4*PHI)


def overlap_bessel(v,flux):
    v=np.asarray(v,complex)
    norm=np.linalg.norm(v)
    if norm<=0 or not np.isfinite(norm):raise ValueError('Invalid finite vector')
    v=v/norm
    ue=np.exp(-8j*PHI*np.asarray(flux,dtype=float)**2)
    return np.vdot(v,exact_infinite_link_matrix_on_source(flux)@(ue*v))


def overlap_padded_matrix(v,flux,padding=16):
    flux=np.asarray(flux,int)
    a=int(flux[0])-padding;b=int(flux[-1])+padding
    r=np.arange(a,b+1)
    d=4*np.eye(len(r),dtype=complex)
    indices=np.arange(len(r)-1)
    d[indices,indices+1]=-2
    d[indices+1,indices]=-2
    gate=expm(-1j*PHI*d)
    f=np.zeros(len(r),complex)
    f[flux-a]=v/np.linalg.norm(v)
    ue=np.exp(-8j*PHI*r.astype(float)**2)
    return np.vdot(f,gate@(ue*f))


def defect_from_overlap(a):
    # Optimizes over a global unimodular Floquet eigenphase.
    return float(np.sqrt(max(0.,2*(1-abs(a)))))


def run():
    ground=ground_run()
    rows=[]
    for row in ground['rows']:
        N=row['support_flux_N']
        with np.load(EVIDENCE/row['file'],allow_pickle=False) as data:
            f=data['rho_gram_factor'][:,0]
            flux=data['flux']
        bessel=overlap_bessel(f,flux)
        padded=overlap_padded_matrix(f,flux,padding=16)
        error=abs(bessel-padded)
        residual=defect_from_overlap(bessel)
        if not(error<1e-12 and 0.03<residual<0.04):
            raise ValueError('Split-step diagnostic failed independent numerical check')
        rows.append({'N':N,'overlap_Bessel_complex':[float(bessel.real),float(bessel.imag)],
                    'optimized_Floquet_eigenphase_defect':residual,
                    'independent_padded_expm_overlap_discrepancy':float(error),
                    'source_distance_to_primitive_ground_upper':row['trace_distance_to_infinite_ground_upper_exact_comparison'],
                    'classification':'FINITE_SOURCE_SPLIT_STEP_U1_DIAGNOSTIC'})
    report={'scope':'SINGLE_PLAQUETTE_U1_SPLIT_STEP_GAUGE_SUBSYSTEM_NOT_FULL_QICT',
            'frozen_factor_order':'exp(-i phi K_B) exp(-i phi K_E)',
            'phi':'pi/63',
            'K_B':'4-2(S+S_dagger)', 'K_E':'8*r^2',
            'unwrapped_H_ground_stationary_for_split_step_numerically':False,
            'numerical_optimized_phase_defect':rows,
            'two_independent_finite_source_overlap_algorithms':True,
            'logical_conclusion':'The primitive Hamiltonian pole and residue cannot be equated with a Floquet pole of the ordered microscopic product.',
            'full_U_int_route_first_implemented':False,
            'physical_Floquet_pole_certified':False}
    (ROOT/'results/PRIMITIVE_SPLIT_STEP_STATIONARITY_CHECK.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

if __name__=='__main__':
    r=run()
    for e in r['numerical_optimized_phase_defect']:
        print(e['N'],e['optimized_Floquet_eigenphase_defect'],e['independent_padded_expm_overlap_discrepancy'])
