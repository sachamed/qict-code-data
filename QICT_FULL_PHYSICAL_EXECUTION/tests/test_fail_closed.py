import pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from stage_guard import guard
from validate_physical_inputs import source_isometry,moment_diagnostic,check_density
import numpy as np
class RefusalTests(unittest.TestCase):
    def test_missing_buffers(self):
        with tempfile.TemporaryDirectory() as path:
            answer=guard("kato",path)
            self.assertEqual(answer["status"],"BLOCKED_MISSING_INPUT")
            self.assertFalse(answer["accepted_for_physical_execution"])
    def test_nonisometric_source(self):
        with self.assertRaises(ValueError): source_isometry(np.zeros((8,6)))
    def test_incorrect_C0(self):
        with self.assertRaises(ValueError): moment_diagnostic(np.zeros((2,6,6)),np.zeros(2))
    def test_negative_density(self):
        with self.assertRaises(ValueError): check_density(np.diag([1.,-1.]),np.eye(2))
    def test_pole_blocked(self):
        with tempfile.TemporaryDirectory() as path:
            self.assertFalse(guard("pole",path)["accepted_for_physical_execution"])
if __name__=="__main__":unittest.main()
