"""Exact input and parity checks for the proposed quark extension."""
from fractions import Fraction as F
import json

# Previously bounded reduced coefficient; no quark mass is used to fix it.
alpha = (
    F("0.156806466326680890086010559318807771034054023636493057035694"),
    F("0.156806466326680890086010890412272039301550341058241868425933"),
)

def logarithm_bounds(x, terms=100):
    x = F(x)
    power = 0
    while x >= 2:
        x /= 2
        power += 1
    def bounded_unit(y):
        z = (y - 1) / (y + 1)
        partial = 2 * sum(z**(2*n+1) / (2*n+1) for n in range(terms))
        remainder = 2*z**(2*terms+1) / ((2*terms+1)*(1-z*z))
        return partial, partial + remainder
    lower, upper = bounded_unit(x)
    lower2, upper2 = bounded_unit(F(2))
    return lower + power*lower2, upper + power*upper2

def required_gap(x):
    lower, upper = logarithm_bounds(x)
    return lower / alpha[1], upper / alpha[0]

def polynomial(d):
    return 2*d*d + d

def representation_cost(y, colour, weak):
    return 3*colour + 4*weak*(weak+1) + polynomial(y)

representations = {
    "Q_L": (F(1), F(4,3), F(1,2)),
    "u_R": (F(4), F(4,3), F(0)),
    "d_R": (F(-2), F(4,3), F(0)),
    "L_L": (F(-3), F(0), F(1,2)),
    "e_R": (F(-6), F(0), F(0)),
    "nu_R": (F(0), F(0), F(0)),
}
costs = {name: str(representation_cost(*values))
         for name, values in representations.items()}
assert costs == {"Q_L":"10", "u_R":"40", "d_R":"10",
                 "L_L":"18", "e_R":"66", "nu_R":"0"}

routes = list(range(1,21,2))
differences = sorted({polynomial(b)-polynomial(a)
                      for a in routes for b in routes if b > a})
assert all(d % 2 == 0 for d in differences)
lower = required_gap(F(1030,53))[0]
upper = required_gap(F(9430,463))[1]
assert 18 < lower < upper < 20
assert not any(lower <= d <= upper for d in differences)
central = required_gap(F(935,47))

# Executable controls for the general cancellation identity.
for c3 in [F(-100), F(0), F(3), F(100), F(1,7)]:
    pd = polynomial(5) + F(4,3)*c3
    ps = polynomial(3) + F(4,3)*c3
    assert pd - ps == 34

print(json.dumps({
    "calculation": "QUARK_COST_AND_PARITY_IDENTITIES_VERIFIED",
    "primitive_representation_costs": costs,
    "required_central_gap_interval": [float(x) for x in central],
    "required_endpoint_envelope": [float(lower), float(upper)],
    "odd_route_defects": routes,
    "admissible_gap_intersection": "empty",
    "physical_mass_matching": "requires the colour-resolved interacting kernel",
}, indent=2))
