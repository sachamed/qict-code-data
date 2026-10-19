#!/usr/bin/env python3
"""Certified primitive one-plaquette infinite-flux limits from exact binary source bits.

The finite source vectors are dyadic rationals when interpreted as stored
IEEE-754 numbers. All Rayleigh/variance/projection inequalities are verified
via fractions, without interpreting a float eigensolver's tiny residual as a
rigorous bound. The physical charged Floquet states are NOT being computed.
"""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from hashlib import sha256
from math import isqrt
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence'
CERTFILE='PRIMITIVE_A7_GROUND_CERTIFICATE.json'


def file_digest(p:Path)->str:
    h=sha256()
    with p.open('rb') as fd:
        for b in iter(lambda:fd.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def exact_dyadic(x):
    if not np.isfinite(x):raise ValueError('Non-finite input')
    return Fraction.from_float(float(x))


def frac_decimal(q:Fraction, rounding:str=ROUND_CEILING, prec:int=48)->str:
    with localcontext() as ctx:
        ctx.prec=prec
        ctx.rounding=rounding
        return str(Decimal(q.numerator)/Decimal(q.denominator))


def sqrt_upper_decimal(q:Fraction, digits:int=18)->str:
    if q<0:raise ValueError('Square root of negative number')
    scale=10**digits
    v=q.numerator*scale*scale
    n=isqrt(v//q.denominator)
    if n*n*q.denominator < v:n+=1
    return f'{n//scale}.{n%scale:0{digits}d}'


def primitive_exact_certificate(flux,amplitudes):
    flux=np.asarray(flux,dtype=np.int64)
    psi=np.asarray(amplitudes,dtype=np.complex128)
    if flux.ndim!=1 or psi.shape!=flux.shape or flux.size<3:
        raise ValueError('Invalid flux dimensions')
    if not np.array_equal(flux,np.arange(flux[0],flux[-1]+1)) or flux[0]!=-flux[-1]:
        raise ValueError('Finite support must be a symmetric consecutive flux interval')
    if not np.isfinite(psi).all() or np.any(psi.imag!=0):
        raise ValueError('Independent real dyadic primitive vector required')
    p={int(r):exact_dyadic(v.real) for r,v in zip(flux,psi)}
    s=sum((v*v for v in p.values()),Fraction(0))
    if s<=0:raise ValueError('Zero state')
    # Integer-valued infinite Hamiltonian H=8r^2+4-2(S+S^*).
    a=sum(((8*r*r+4)*v*v for r,v in p.items()),Fraction(0))
    a-=4*sum((p[r]*p.get(r+1,Fraction(0)) for r in p),Fraction(0))
    hp={r:(8*r*r+4)*p.get(r,Fraction(0))
          -2*p.get(r-1,Fraction(0))-2*p.get(r+1,Fraction(0))
        for r in range(int(flux[0])-1,int(flux[-1])+2)}
    b=sum((v*v for v in hp.values()),Fraction(0))
    variance_numerator=b*s-a*a
    if variance_numerator<0:raise ValueError('Exact variance negative: implementation error')
    if not a<8*s:raise ValueError('Rayleigh energy not below excited-state threshold 8')
    mu=a/s
    # Rayleigh quotient/Temple and spectral overlap are EXACT rational bounds.
    ground_lower=mu-variance_numerator/(s*(8*s-a))
    weight_exc_upper=variance_numerator/((8*s-a)**2)
    if weight_exc_upper>=1:raise ValueError('No nontrivial overlap certificate')
    with localcontext() as ctx:
        ctx.prec=32
        mu_display=float(mu)
        var_display=float(variance_numerator/(s*s))
    return {
      'support_flux_N':int(flux[-1]),'component_count':len(p),
      'operator':'8*r^2+4-2*(S+S_dagger) on ell2(Z)',
      'coefficient_semantics':'stored finite binary64 input values treated as exact dyadic numbers',
      'ground_energy_lower_rational_outward':frac_decimal(ground_lower,ROUND_FLOOR),
      'ground_energy_upper_rational_outward':frac_decimal(mu,ROUND_CEILING),
      'excited_spectrum_lower_exact':8,
      'energy_gap_lower_rational_outward':frac_decimal(8-mu,ROUND_FLOOR),
      'trace_distance_to_infinite_ground_upper_exact_comparison':sqrt_upper_decimal(weight_exc_upper),
      'orthogonal_weight_upper_exact_comparison':frac_decimal(weight_exc_upper,ROUND_CEILING),
      'temple_enclosure_width_upper':frac_decimal(mu-ground_lower,ROUND_CEILING),
      'exact_checks':{'variance_nonnegative':variance_numerator>=0,
                      'rayleigh_below_excited_threshold':a<8*s,
                      'nontrivial_overlap':weight_exc_upper<1,
                      'source_norm_positive':s>0},
      'diagnostics_float64':{'rayleigh':mu_display,'variance':var_display},
      'physical_interacting_KATO_BUFFER':False
    }



def exact_pairwise_density_distance_upper(first, second, digits=18):
    """Exact two-ray overlap in one common ell2(Z) flux basis."""
    p1={int(r):exact_dyadic(x.real) for r,x in first}
    p2={int(r):exact_dyadic(x.real) for r,x in second}
    n1=sum((x*x for x in p1.values()),Fraction(0))
    n2=sum((x*x for x in p2.values()),Fraction(0))
    dot=sum((v*p2.get(r,Fraction(0)) for r,v in p1.items()),Fraction(0))
    d2=1-dot*dot/(n1*n2)
    if not 0<=d2<=1:raise ValueError('Exact fidelity outside [0,1]')
    return sqrt_upper_decimal(d2,digits)

def run():
    cert=json.loads((EVIDENCE/CERTFILE).read_text())
    if cert.get('physical_full_interacting_Floquet_Kato') is not False:
        raise ValueError('Primitive state certificate mislabeled')
    rows=[]
    rays=[]
    for e in cert['cutoffs']:
        p=EVIDENCE/e['file']
        if file_digest(p)!=e['sha256']:raise ValueError(f'Source SHA mismatch {p.name}')
        with np.load(p,allow_pickle=False) as z:
            flux=z['flux']; factors=z['rho_gram_factor'][:,0]
            row=primitive_exact_certificate(flux, factors)
            rays.append(list(zip(flux, factors)))
        if row['support_flux_N']!=e['N']:raise ValueError('Source cutoff mismatch')
        row['file']=e['file'];row['sha256']=e['sha256']
        rows.append(row)
    assert [x['support_flux_N'] for x in rows]==[8,12,16,20]
    # D(phi_N,phi_M) <= D(phi_N,phi_inf)+D(phi_inf,phi_M).
    pairwise=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            upper=(Decimal(rows[i]['trace_distance_to_infinite_ground_upper_exact_comparison'])+
                   Decimal(rows[j]['trace_distance_to_infinite_ground_upper_exact_comparison']))
            direct=exact_pairwise_density_distance_upper(rays[i],rays[j])
            if Decimal(direct)>upper:
                raise ValueError('Certified infinite-ground upper bound contradicts direct exact overlap')
            pairwise.append({'cutoffs':[rows[i]['support_flux_N'],rows[j]['support_flux_N']],
                             'trace_distance_direct_dyadic_upper':direct,
                             'trace_distance_pairwise_upper_from_infinite_ground':str(upper),
                             'common_embedding':'matching integer electric-flux labels in ell2(Z)'})
    report={'classification':'RIGOROUS_RATIONAL_PRIMITIVE_INFINITE_FLUX_CERTIFICATE',
        'model':'single Abelian plaquette unwrapped H_1; not QICT full Floquet/Kato',
        'original_primitive_certificate_sha256':file_digest(EVIDENCE/CERTFILE),
        'spectral_argument':'H>=8r^2; minmax lambda_1>=8; E0<=4; compact resolvent and positivity improving imply unique ground state. Temple lower bound and spectral second moment bound are exact for dyadic trial vectors.',
        'rows':rows,'pairwise_bounds':pairwise,'physical_charged_kato_completed':False,
        'quantitative_continuum_regulator_result':'four source-backed finite vectors rigorously enclose one infinite-flux primitive ground state',
        'tail_certified_for_full_interacting_QICT':False}
    result=ROOT/'results/PRIMITIVE_INFINITE_VOLUME_RATIONAL_CERTIFICATE.json'
    result.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

if __name__=='__main__':
    report=run()
    for row in report['rows']:
        print(row['support_flux_N'],row['ground_energy_lower_rational_outward'],
              row['ground_energy_upper_rational_outward'],
              row['trace_distance_to_infinite_ground_upper_exact_comparison'])
