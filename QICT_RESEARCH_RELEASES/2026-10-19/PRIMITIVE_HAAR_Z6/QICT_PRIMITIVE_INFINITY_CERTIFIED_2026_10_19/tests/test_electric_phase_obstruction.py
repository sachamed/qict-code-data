import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from electric_phase_obstruction import exact_obstruction,phase_one,trace_distance_nested,identity_h
class ElectricPhaseTests(unittest.TestCase):
    def test_integer_phase_periodicity(self):
        self.assertTrue(all(identity_h(n,eta) for n in range(-8,9) for eta in [-5,-1,0,3,7]))
    def test_flux_eigenphase(self):
        self.assertTrue(phase_one(21))
        self.assertFalse(phase_one(7))
    def test_non_cauchy_trace_distance(self):
        self.assertEqual(str(trace_distance_nested(63)),'6/13')
        self.assertEqual(str(trace_distance_nested(126)),'12/25')
        self.assertTrue(exact_obstruction()['not_trace_norm_cauchy'])
if __name__=='__main__':unittest.main()
