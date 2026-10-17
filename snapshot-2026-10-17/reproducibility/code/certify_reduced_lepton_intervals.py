#!/usr/bin/env python3
"""Enclose reduced CAR/BCC lepton invariants from the rational rotor certificate.

Uses the theorem-level SU(2) Wilson interval as the input. These are intervals
for the reduced construction, not physical interacting lepton mass uncertainties.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, localcontext
from mpmath import iv


def endpoints(x):
    text = str(x)
    matched = re.fullmatch(r'\[([^,]+), ([^\]]+)\]', text)
    if matched is None:
        raise ValueError(f'Unrecognized interval: {text}')
    return Decimal(matched.group(1)), Decimal(matched.group(2))


def outward(x, places=18):
    low, high = endpoints(x)
    quantum = Decimal(1).scaleb(-places)
    with localcontext() as ctx:
        ctx.prec = 90
        return [str(low.quantize(quantum, rounding=ROUND_FLOOR)),
                str(high.quantize(quantum, rounding=ROUND_CEILING))]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--rotor', type=Path, required=True)
    pa.add_argument('--out', type=Path, required=True)
    args = pa.parse_args()
    source = json.loads(args.rotor.read_text())
    assert source['all_rational_checks_pass']
    source_cert = max(source['certificates'],key=lambda c:c['N'])
    assert source_cert['N'] >= 24
    iv.dps = 75
    lo,hi = source_cert['wilson_expectation_interval']
    wilson = iv.mpf([lo,hi]); one = iv.mpf(1)
    bcc = iv.gamma(one/4)**4/(32*iv.pi**3)
    phi = iv.pi/63
    afac = bcc*iv.sqrt(wilson)*iv.cos(phi)
    assert endpoints(afac)[0]>0 and endpoints(afac)[1]<1
    aconn = afac-iv.log(1-afac**3)/96
    e=one; mu=iv.exp(34*aconn); tau=iv.exp(52*aconn)
    koide=(e+mu+tau)/(iv.sqrt(e)+iv.sqrt(mu)+iv.sqrt(tau))**2
    data={
      'classification':'OUTWARD_INTERVAL_REDUCED_CAR_BCC_QUANTITIES',
      'scope':'Reduced linked-cluster and receiver route-to-flux invariants; no claim of physical pole-mass intervals',
      'source':'CERTIFIED_SU2_ROTOR.json',
      'rotor_truncation':source_cert['N'],
      'rotor_wilson_enclosure':source_cert['wilson_expectation_interval'],
      'R_BCC_interval':outward(bcc,27),
      'alpha_fac_interval':outward(afac,21),
      'alpha_conn_interval':outward(aconn,21),
      'mu_over_e_reduced_interval':outward(mu,15),
      'tau_over_mu_reduced_interval':outward(tau/mu,15),
      'tau_over_e_reduced_interval':outward(tau,15),
      'koide_reduced_interval':outward(koide,18),
      'arithmetic':'mpmath.iv interval arithmetic at 75 decimal places, with outward Decimal rounding',
      'ratio_exponents':{'mu_over_e':34,'tau_over_mu':18,'tau_over_e':52},
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2)+'\n')
    print('certified reduced mu/e',data['mu_over_e_reduced_interval'])
    print('certified reduced tau/e',data['tau_over_e_reduced_interval'])

if __name__ == '__main__':main()
