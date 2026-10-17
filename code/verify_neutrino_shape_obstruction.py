#!/usr/bin/env python3
"""Finite QICT neutrino-shape test; measurements are comparison targets, not fitted QICT inputs."""
import json
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 55
D = lambda x: Decimal(str(x))
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "analytical_certificates" / "NEUTRAL_SHAPE_SHARP_DISTANCE.json"

def diagnostic(s, ref):
    a,b,c = s
    assert 0 < a < b < c
    g21,g31 = b*b-a*a,c*c-a*a
    d21,d31 = D(ref["delta_m21_eV2"]),D(ref["delta_m3l_eV2"])
    r,r0 = d21/d31,g21/g31
    assert 0 < r < r0 < 1
    eps = (g21-r*g31)/(2*(a+b+r*(c-a)))
    assert 0 < eps < (b-a)/2
    t = (a+eps,b-eps,c+eps)
    actual = (t[1]**2-t[0]**2)/(t[2]**2-t[0]**2)
    assert abs(actual-r) < D("1e-45")
    small = eps-D("0.000001")
    if small <= 0: small = eps/2
    rmin = ((b-small)**2-(a+small)**2)/((c+small)**2-(a+small)**2)
    assert rmin > r
    out = {
      "finite_ratio":str(r0),"external_ratio":str(r),
      "exact_minimum_additive_operator_norm":str(eps),
      "minimum_fraction_of_largest_singular_value":str(eps/c),
      "necessary_kinetic_norm_deviation_if_E_zero":str(eps/(c+eps)),
      "joint_necessary_condition":"||E|| >= max(0, epsilon*(1-d)-c*d), d=||K-I||<1",
      "saturating_singular_values_mathematical_witness":list(map(str,t)),
      "verified_witness_ratio":str(actual),
      "interpretation":"A necessary finite spectral compatibility constraint; the witness is not a QICT self-energy."
    }
    fields = ("delta_m21_sigma_minus_eV2","delta_m21_sigma_plus_eV2",
              "delta_m3l_sigma_minus_eV2","delta_m3l_sigma_plus_eV2")
    if all(k in ref for k in fields):
        s21 = (D(ref[fields[0]])+D(ref[fields[1]]))/2
        s31 = (D(ref[fields[2]])+D(ref[fields[3]]))/2
        assert s21>0 and s31>0
        w21,w31 = 1/(s21*s21),1/(s31*s31)
        x = (w21*g21*d21+w31*g31*d31)/(w21*g21*g21+w31*g31*g31)
        chi2 = ((x*g21-d21)/s21)**2 + ((x*g31-d31)/s31)**2
        out.update({"profiled_squared_scale":str(x),
           "chi2_diagonal_gaussian_surrogate":str(chi2),
           "statistic_status":"Two independent symmetric-error Gaussian marginals; NOT the correlated NuFIT likelihood."})
    return out

def main():
    b = json.loads((DATA/"FINITE_FLAVOR_BENCHMARKS.json").read_text())
    refs = json.loads((DATA/"EXPERIMENTAL_REFERENCE_VALUES_2026.json").read_text())
    singular = tuple(map(D,b["neutral_bare_takagi_singular_values"]))
    cases = {
      "NuFIT_6_0_NO":diagnostic(singular,refs["pmns_normal_ordering"]),
      "NuFIT_6_1_NO_IC23_noSK":diagnostic(singular,refs["pmns_normal_ordering_nufit61"])
    }
    assert abs(D(cases["NuFIT_6_0_NO"]["chi2_diagonal_gaussian_surrogate"])-D("6814.89273987846")) < D("1e-6")
    assert abs(D(cases["NuFIT_6_1_NO_IC23_noSK"]["exact_minimum_additive_operator_norm"])-D("0.0322782819468144")) < D("1e-12")
    payload={"classification":"FINITE_NEUTRAL_DIAGNOSTIC_NOT_PHYSICAL_POLE_PREDICTION",
       "no_new_fitted_QICT_parameters":True,
       "finite_takagi_values":list(map(str,singular)),"cases":cases}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print("PASS: NuFIT6.0 chi2 =",cases["NuFIT_6_0_NO"]["chi2_diagonal_gaussian_surrogate"])
    print("PASS: NuFIT6.1 epsilon =",cases["NuFIT_6_1_NO_IC23_noSK"]["exact_minimum_additive_operator_norm"])

if __name__ == "__main__":
    main()
