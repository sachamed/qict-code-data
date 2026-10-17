#!/usr/bin/env python3
"""Small reproducibility calculation for flavor correspondence inequalities.

Uses decimal arithmetic on the displayed finite source values.  The outputs
are mathematical diagnostics, not renormalized QICT mass predictions.
"""
from decimal import Decimal as D, getcontext
from pathlib import Path
import json
import math

getcontext().prec = 65


def neutral_minimum():
    s1, s2, s3 = map(D, ('0.1895171061', '0.2674032426', '0.4706252560'))
    r = D('7.49e-5') / D('2.534e-3')
    a2, a3 = s2*s2-s1*s1, s3*s3-s1*s1
    gap = abs(a2-r*a3)
    lower = ((s3*s3 + gap/(2*(1+r))).sqrt() - s3)
    assert (1+r)*(4*s3*lower+2*lower*lower) >= gap-D('1e-57')
    assert (1+r)*(4*s3*(lower-D('1e-12'))+2*(lower-D('1e-12'))**2) < gap
    return {'finite_singular_values': list(map(str,(s1,s2,s3))),
            'experimental_gap_ratio_reference': str(r),
            'finite_gap_ratio': str(a2/a3),
            'minimum_correction_operator_norm': str(lower),
            'relative_to_largest_singular_value': str(lower/s3)}


def down_minimum():
    s1,s2 = D('0.015499504'),D('0.576934470')
    r = D('4.69') / D('92.74')
    lower = (r*s2-s1)/(1+r)
    assert (s1+lower)/(s2-lower) >= r-D('1e-59')
    return {'finite_singular_values':[str(s1),str(s2)],
            'short_distance_ratio_input':str(r),
            'minimum_correction_operator_norm':str(lower),
            'relative_to_second_singular_value':str(lower/s2)}


def lepton_residuals():
    rm = D('206.7312943550374075')
    rt = D('3477.0386539527347655')
    # Reference ratios are displayed to limited accuracy in the prior dataset.
    obs_mu = D('206.768282708')
    obs_tau = D('3477.365262')
    qmu = (obs_mu/rm).ln()
    qtau = (obs_tau/rt).ln()
    lo = min(D(0),qmu,qtau)
    hi = max(D(0),qmu,qtau)
    return {'reference_mu_e':str(obs_mu),'reference_tau_e':str(obs_tau),
            'reduced_mu_e':str(rm),'reduced_tau_e':str(rt),
            'log_residual_mu':str(qmu), 'log_residual_tau':str(qtau),
            'minimum_empirical_max_log_mismatch_after_common_normalization':str((hi-lo)/2)}


def rouche_and_ratio_examples():
    # Exact scalar poles for Sigma(z)=a+b(z-m0): the derivative is nonzero.
    a, b, m0, r = D('0.006'),D('0.05'),D('1.4'),D('0.2')
    root = m0+a/(1-b)
    eps = abs(a)+abs(b)*r
    assert eps < r
    assert abs(root-m0) <= eps
    betae, betaf = D('0.01'),D('0.02')
    for xe,xf in [(D('0.01'),D('0.02')),(D('-0.01'),D('-0.02')),
                  (D('0'),D('0'))]:
        ratio = (1+xf)/(1+xe)
        assert (1-betaf)/(1+betae) <= ratio <= (1+betaf)/(1-betae)
    return {'linear_self_energy_root':str(root), 'radius':str(r),
            'sup_self_energy_bound':str(eps), 'root_displacement':str(abs(root-m0))}


def main():
    output = {
        'calculation_type':'exact_decimal_arithmetic_on_specified_rounded_inputs',
        'physical_status':'finite-representative necessary correction bounds and external residual diagnostics',
        'neutral':neutral_minimum(),
        'down':down_minimum(),
        'charged_lepton_residuals':lepton_residuals(),
        'analytic_pole_inequality_test':rouche_and_ratio_examples(),
        'route_costs':{'P_e':55,'P_mu':21,'P_tau':3,
                       'P_e_minus_P_mu':34,'P_e_minus_P_tau':52},
    }
    assert output['route_costs']['P_e_minus_P_mu'] == 55-21
    assert output['route_costs']['P_e_minus_P_tau'] == 55-3
    target = Path(__file__).resolve().parent.parent/'results'/'FLAVOR_CORRESPONDENCE_BOUNDS.json'
    target.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__ == '__main__':
    main()
