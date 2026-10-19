"""Independent algebraic checks of quantitative contour perturbation theorem."""
import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quantitative_stability import regression,quantitative_projector_bound,compressed_residue_bound
class SpectralStabilityTests(unittest.TestCase):
    def test_spectral_gap_projector_bound(self):
        for eps in (0.01,0.04,0.07):
            x=regression(eps,4)
            self.assertLess(x['projector_difference_norm'],x['rigorous_projector_difference_upper'])
            self.assertGreater(x['smallest_compressed_residue_eigenvalue'],.99)
    def test_nonzero_gap_required(self):
        with self.assertRaises(ValueError): quantitative_projector_bound(0.1,2*math.pi,0)
    def test_no_negative_error(self):
        with self.assertRaises(ValueError): compressed_residue_bound(-1,0,2*math.pi,1)
    def test_exact_formula_against_analytic_rotation(self):
        eps=.04;x=regression(eps,3)
        # Same symmetric operator across m cells. For the two-phase matrix
        # U0=diag(1,i), the commutator rotation gives exact norm
        # |1-i| sin eps, while spectral projectors differ by sin eps.
        self.assertAlmostEqual(x['operator_difference_norm'],math.sqrt(2)*math.sin(eps),places=12)
        self.assertAlmostEqual(x['projector_difference_norm'],math.sin(eps),places=12)
        self.assertAlmostEqual(x['smallest_compressed_residue_eigenvalue'],math.cos(eps)**2,places=12)
if __name__=='__main__':unittest.main()
