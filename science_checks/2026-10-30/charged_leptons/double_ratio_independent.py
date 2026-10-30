#!/usr/bin/env python3
"""Independent QICT charged-lepton reference test; experimental targets, not predicted masses."""
import json
from decimal import Decimal, localcontext
from pathlib import Path


def compute(reference):
    with localcontext() as ctx:
        ctx.prec = 65
        m = {f: Decimal(reference['masses_MeV'][f]['value']) for f in ('e','mu','tau')}
        a = Decimal(reference['alpha_conn'])
        dm = (m['mu']/m['e']).ln() - 34*a
        dt = (m['tau']/m['e']).ln() - 52*a
        invariant = 26*dm - 17*dt
        span = max(Decimal(0), dm, dt) - min(Decimal(0), dm, dt)
        return {'delta_mu':str(dm), 'delta_tau':str(dt), 'D':str(invariant),
                'rho_oscillation_central':str(span),
                'max_factor_ratio_minus_one':str(span.exp()-1),
                'classification':'experimental targets; no physical transfer prediction'}


def main():
    source = Path(__file__).with_name('reference_2026.json')
    result = compute(json.loads(source.read_text(encoding='utf-8')))
    assert abs(Decimal(result['D'])-Decimal('0.00305472016780177433988')) < Decimal('1e-22')
    assert abs(Decimal(result['delta_mu'])-Decimal('0.000178903948541391930196')) < Decimal('1e-24')
    assert abs(Decimal(result['delta_tau'])-Decimal('0.000093928382016142108542')) < Decimal('1e-24')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
