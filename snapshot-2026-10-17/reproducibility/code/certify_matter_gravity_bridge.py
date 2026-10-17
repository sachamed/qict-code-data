#!/usr/bin/env python3
"""Exact clock-independent matching identities and live input-availability audit.

No physical QICT masses, gamma_R or cosmological term are inferred from
experimental data, reduced benchmarks, or synthetic self-tests.
"""
from pathlib import Path
import json
import sympy as sp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def check_equal(label, expression, checks):
    delta = sp.factor(expression)
    if delta != 0:
        raise AssertionError(f'{label}: {delta}')
    checks.append(label)

def jacobian(n):
    # Effective logarithmic map, not the complete microscopic parameter map.
    return sp.Matrix([[2, -1, 0] + [0]*n,
                      [-2, -1, 1] + [0]*n] +
                     [[-1, 0, 0] + [int(j == i) for j in range(n)]
                      for i in range(n)])

def main():
    checks=[]
    tau,gamma,epsilon0,eps,c,hbar=sp.symbols('tau gamma epsilon0 eps c hbar', nonzero=True)
    mass=hbar*eps/(c**2*tau)
    G=c**5*tau**2/(8*sp.pi*hbar*gamma)
    cosm=epsilon0/(gamma*c**2*tau**2)
    alpha=G*mass**2/(hbar*c)
    lambdaC=hbar/(mass*c)
    check_equal('matter-gravity alpha invariant',8*sp.pi*alpha-eps**2/gamma,checks)
    check_equal('cosmological Compton invariant',cosm*lambdaC**2-epsilon0/(gamma*eps**2),checks)
    check_equal('cosmological gravity invariant',8*sp.pi*hbar*G*cosm/c**3-epsilon0/gamma**2,checks)
    check_equal('three-invariant triangle',8*sp.pi*alpha*cosm*lambdaC**2-8*sp.pi*hbar*G*cosm/c**3,checks)
    check_equal('common physical scale cancels mass ratios', (hbar*eps/(c**2*tau))/(hbar*sp.Symbol('eta',nonzero=True)/(c**2*tau))-eps/sp.Symbol('eta',nonzero=True),checks)
    ranks={}
    for n in (1,3,6,12):
        M=jacobian(n)
        null=sp.Matrix([1,2,4]+[1]*n)
        assert M*null == sp.zeros(n+2,1)
        assert M.rank() == n+2
        assert sp.Matrix.vstack(M,sp.Matrix([[1]+[0]*(n+2)])).rank()==n+3
        checks.extend([f'Jacobian rank N={n}',f'null direction N={n}',f'independent clock rank N={n}'])
        ranks[str(n)]={'shape':list(M.shape),'rank':M.rank(),
                       'null_direction':[int(x) for x in null], 'rank_with_independent_clock':n+3}
    k,b1,b2,w,q=sp.symbols('k b1 b2 w q',positive=True)
    F1=k*q*q-b1*w*w
    F2=k*q*q-b2*w*w
    check_equal('identical static TT family',F1.subs(w,0)-F2.subs(w,0),checks)
    assert sp.simplify(k/b1-k/b2)!=0
    assert sp.simplify(1/b1-1/b2)!=0
    checks+=['different TT pole velocity symbols','different inverse pole residue symbols']

    datafile=ROOT/'data/SPECTRAL_GRAVITY_SELFTEST_INPUT.npz'
    required={'K0','K1_vol','K1_q','K1_mq','K2_qmq','N_cells','qell','continuum_density_factor'}
    with np.load(datafile, allow_pickle=False) as d:
        keys=set(d.files)
        assert required <= keys
        spectral={'type':'finite_source_operator_selftest','relative_path':str(datafile.relative_to(ROOT)),
                  'keys':sorted(keys), 'qell':float(d['qell']),
                  'sampled_spatial_momenta':1,
                  'provides_dynamical_frequency_derivatives':False,
                  'can_certify_gamma_R_continuum':False,
                  'can_certify_massless_graviton':False}
    checks.append('actual gravitational benchmark schema verified')

    auditfile=ROOT/'results/physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json'
    audit=json.loads(auditfile.read_text())
    expected=audit['continuum_sequence_inputs']
    physical_checks=audit['physical_contract_checks']
    assert set(expected)==set(physical_checks)
    missing=[key for key in expected if physical_checks[key] is not True]
    assert missing
    assert not audit['physical_G2_promoted']
    checks.append('physical G2 production-class status verified')
    result={
      'classification':'EXACT_CROSS_SECTOR_IDENTITIES_AND_INPUT_READINESS',
      'proof_scope':'Algebraic identities and effective parameter-map ranks; no physical mass or gravitational completion',
      'symbolic_checks_passed':len(checks), 'checks':checks,
      'exact_invariants':{
         '8pi G m_f^2/(hbar c)':'epsilon_f^2/gamma_R',
         'Lambda (hbar/(m_f c))^2':'epsilon_0/(gamma_R epsilon_f^2)',
         '8pi hbar G Lambda/c^3':'epsilon_0/gamma_R^2'
      },
      'identifiability_n_mass_channels':ranks,
      'static_TT_nonidentifiability':{
         'shared_static_kernel':'kappa q^2',
         'Lorentzian_inverse_kernels':'kappa q^2 - b omega^2 (b>0)',
         'wave_speed_squared':'kappa/b',
         'pole_residue_magnitude':'1/b',
         'interpretation':'Counterexample family only; does not realize QICT spin-2 sector'},
      'gravity_input_available':spectral,
      'charged_physical_inputs':{'complete':len(expected)-len(missing),'required':len(expected),
                                 'missing':missing,'physical_G2_promoted':False},
      'physical_masses_computed':False,
      'physical_G_Lambda_computed':False,
      'number_of_fundamental_free_parameters_established':False,
      'calibration_policy':'No experimental input enters this certificate.'
    }
    out=ROOT/'results/MATTER_GRAVITY_BRIDGE_EXACT_CERTIFICATE.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(f"PASS {len(checks)} symbolic and input checks; {len(missing)}/{len(expected)} missing physical charged input classes")
    print(out)
if __name__=='__main__':main()
