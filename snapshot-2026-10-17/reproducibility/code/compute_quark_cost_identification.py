from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT
sys.path.insert(0,str(SOURCE/'reproducibility/code'))
from quantitative_comparison import I,exp_positive
from test_reduced_hierarchy import rational_log
cert=SOURCE/'reproducibility/results/quantitative_comparison/QUANTITATIVE_COMPARISON.json'
data=json.loads(cert.read_text());alpha=I(*data['rotor_cutoff_stability'][-1]['alpha_interval'])
reference=F('93.5')/F('4.70');box=(F('92.7')/F('4.77'),F('94.3')/F('4.63'))
gap=rational_log(reference)/alpha
gapbox=(rational_log(box[0])/alpha,rational_log(box[1])/alpha)
def cost(y,c3,j):return 3*c3+4*j*(j+1)+2*y*y+y
representations={'Q_L':(F(1),F(4,3),F(1,2)),'u_R':(F(4),F(4,3),F(0)),
 'd_R':(F(-2),F(4,3),F(0)),'L_L':(F(-3),F(0),F(1,2)),
 'e_R':(F(-6),F(0),F(0)),'nu_R':(F(0),F(0),F(0))}
costs={name:str(cost(*v)) for name,v in representations.items()}
P=lambda d:2*d*d+d
routes=list(range(1,21,2));deltas=sorted({P(b)-P(a) for a in routes for b in routes if b>a})
assert all(x%2==0 for x in deltas)
assert not any(gapbox[0].a<=x<=gapbox[1].b for x in deltas)
for c3 in [F(-100),F(0),F(3),F(100),F(1,7)]:
 assert ((F(P(5))+4*c3/3)-(F(P(3))+4*c3/3))==34
 # Equality after cancellation is exact; these examples are executable controls.
finite=json.loads((SOURCE/'reproducibility/data/FINITE_FLAVOR_BENCHMARKS.json').read_text())
sd=finite['receiver_singular_values']['d'];finite_ratio=sd[1]/sd[0]
prediction=exp_positive(34*alpha);log_correction=rational_log(reference)-34*alpha
result={'calculation':'EXECUTED_QUARK_EXTENSION_DIAGNOSTIC',
 'alpha_input_sha256':hashlib.sha256(cert.read_bytes()).hexdigest(),
 'additive_colour_result':'A common Casimir contribution cancels identically for every coefficient.',
 'primitive_representation_electric_costs':costs,
 'strange_down_reduced_ratio_interval':prediction.pair(),
 'reference_central_exact':str(reference),'reference_individual_90_percent_envelope':[str(x) for x in box],
 'required_cost_difference_interval':gap.pair(),
 'required_cost_envelope':[gapbox[0].pair(),gapbox[1].pair()],
 'positive_C21_route_defects':routes,'all_positive_oriented_route_cost_differences':deltas,
 'integer_parity_test':'No allowed even cost difference belongs to the reference envelope.',
 'required_noncommon_log_correction_interval':log_correction.pair(),
 'correction_interpretation':'The displayed correction is inferred from the comparison and is not inserted into the model.',
 'finite_down_singular_ratio':finite_ratio,
 'physical_quark_matching':'The colour-resolved interacting chirality-changing kernel and its mass-scheme matching are not evaluated by this diagnostic.'}
(ROOT/'reproducibility/results/quark_cost_identification/QUARK_EXTENSION_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='all_positive_oriented_route_cost_differences'},indent=2))
