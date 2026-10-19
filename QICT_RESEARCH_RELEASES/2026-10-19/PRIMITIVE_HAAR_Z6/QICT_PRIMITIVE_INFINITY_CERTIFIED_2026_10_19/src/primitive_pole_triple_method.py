#!/usr/bin/env python3
"""Three mutually cross-checked evaluations of the Abelian primitive local pole.

One algorithm yields rigorous rational intervals from an infinite-operator
residual. The other two independent finite-dimensional methods (Jacobi
spectral diagonalization and continued-fraction Schur) are float64 diagnostics.
This is NOT the physical QICT charged or dark Floquet pole.
"""
from __future__ import annotations
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from primitive_infinite_volume_certificate import (
    ROOT,EVIDENCE,exact_dyadic,file_digest,frac_decimal,run as primitive_run)


def schur_tail_and_derivative(x,N):
    """Even-parity folded half-line Schur complement, k=N,...,1."""
    f=0.0;df=0.0
    for k in range(N,0,-1):
        den=x-(8*k*k+4)-4*f
        if den==0:raise ValueError('Pole of eliminated complementary tail')
        new=1/den
        dnew=-(1-4*df)*new*new
        f,df=new,dnew
    return f,df


def schur_equation(z,N):
    f,_=schur_tail_and_derivative(z,N)
    return z-4-8*f


def schur_pole_and_residue(N):
    e=float(brentq(lambda x:schur_equation(x,N),1.0,3.99,xtol=4e-15))
    _,der=schur_tail_and_derivative(e,N)
    residue=1/(1-8*der)
    return e,residue


def jacobi_pole_and_residue(N):
    """Parity-even sector: e0, (e±1)/sqrt2, ... (e±N)/sqrt2."""
    diag=np.array([8*k*k+4 for k in range(N+1)],float)
    off=np.array([-2*np.sqrt(2)]+[-2.]*(N-1),float)
    # Full-spectrum STEV avoids the default selected-eigenpair bisection
    # tolerance degradation at large diagonal dynamic range.
    w,v=eigh_tridiagonal(diag,off,lapack_driver='stev')
    return float(w[0]),float(abs(v[0,0])**2)


def create_pole_certificate():
    trial=primitive_run()
    strongest=trial['rows'][-1]
    path=EVIDENCE/strongest['file']
    with np.load(path,allow_pickle=False) as d:
        p=np.asarray(d['rho_gram_factor'][:,0],complex)
        flux=np.asarray(d['flux'],int)
    wave={int(r):exact_dyadic(x.real) for r,x in zip(flux,p)}
    s=sum((v*v for v in wave.values()),Fraction(0))
    ztrial=wave[0]*wave[0]/s
    dist=Fraction(Decimal(strongest['trace_distance_to_infinite_ground_upper_exact_comparison']))
    low=ztrial-dist
    high=ztrial+dist
    # Variational analytical mass-localization inequality from H >=8r^2.
    # For normalized phi0, P(|r|>=1) <= E0/8 <= trial_mu/8.
    mu_upper=Fraction(Decimal(strongest['ground_energy_upper_rational_outward']))
    coarse=1-mu_upper/8
    if not 0<coarse<=low<=high<1:
        raise ValueError('Inconsistent local pole and Riesz residue enclosures')
    e_low=Decimal(strongest['ground_energy_lower_rational_outward'])
    e_high=Decimal(strongest['ground_energy_upper_rational_outward'])
    comparisons=[]
    for N in (8,12,16,20,28,40):
        ef,rf=schur_pole_and_residue(N)
        ej,rj=jacobi_pole_and_residue(N)
        if not (abs(ef-ej)<1e-12 and abs(rf-rj)<1e-12):
            raise ValueError('Independent primitive numerical algorithms disagree')
        if not (abs(ef-float(e_high))<3e-12 and abs(rf-float(ztrial))<3e-12):
            raise ValueError('Numerical pole not consistent with rigorous trial certificate')
        comparisons.append({'flux_cutoff':N,'continued_fraction_pole_float64':ef,
          'continued_fraction_residue_float64':rf,'jacobi_pole_float64':ej,
          'jacobi_residue_float64':rj,
          'pole_methods_difference':abs(ef-ej),'residue_methods_difference':abs(rf-rj)})
    certificate={
        'classification':'INFINITE_FLUX_SINGLE_PLAQUETTE_PRIMITIVE_LOCAL_POLE_CERTIFICATE',
        'operator':'H=8r^2+4-2(S+S_dagger), r in Z',
        'local_source':'delta_r0',
        'rigorously_isolated_hamiltonian_ground_pole':True,
        'isolating_contour':'|z|=6',
        'minimum_contour_spectral_distance_lower':2,
        'excited_spectrum_lower':8,
        'ground_energy_lower_exact_rational_outward':strongest['ground_energy_lower_rational_outward'],
        'ground_energy_upper_exact_rational_outward':strongest['ground_energy_upper_rational_outward'],
        'spectral_residue_at_site_zero_lower_exact_rational_outward':frac_decimal(low,ROUND_FLOOR),
        'spectral_residue_at_site_zero_upper_exact_rational_outward':frac_decimal(high,ROUND_CEILING),
        'residue_coarse_analytic_lower_from_H_ge_8r2':frac_decimal(coarse,ROUND_FLOOR),
        'residue_lower_from_E0_le_4':0.5,
        'local_resolvent_definition':'G_00(z)=<delta0,(z-H)^(-1)delta0>; residue at E0 is |<delta0,ground>|^2',
        'three_calculations':{
            'exact_dyadic_trial_Temple_and_residue_enclosure':'CERTIFIED_FOR_PRIMITIVE_ONLY',
            'Jacobi_even_parity_tridiagonal':'FLOAT64_DIAGNOSTIC',
            'Schur_continued_fraction':'FLOAT64_DIAGNOSTIC',
            'cutoffs':comparisons},
        'original_primitive_npz_sha256':file_digest(path),
        'physical_QICT_charged_or_dark_pole':False,
        'physical_gev_mass':None,
        'full_interacting_C10':False}
    (ROOT/'results/PRIMITIVE_LOCAL_RESOLVENT_POLE_CERTIFICATE.json').write_text(
        json.dumps(certificate,indent=2,sort_keys=True)+'\n')
    return certificate

if __name__=='__main__':
    c=create_pole_certificate()
    print('E0',c['ground_energy_lower_exact_rational_outward'],c['ground_energy_upper_exact_rational_outward'])
    print('Z0',c['spectral_residue_at_site_zero_lower_exact_rational_outward'],
           c['spectral_residue_at_site_zero_upper_exact_rational_outward'])
