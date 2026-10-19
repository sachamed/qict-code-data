"""Checks that published B0 evidence is authentic and not promoted to Kato L=24 physical buffers."""
import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from reconstruct_b0_inputs import reconstruct
class OriginalEvidence(unittest.TestCase):
    def test_b0_cutoff_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            result=reconstruct(pathlib.Path(td))
            self.assertEqual(result["status"],"BLOCKED_MISSING_INPUT")
            self.assertEqual(len(result["factorized_b0_artifacts"]),6)
            self.assertFalse(result["physical_KATO_BUFFER_0_to_3_created"])
            for r in result["factorized_b0_artifacts"]:
                with np.load(pathlib.Path(td)/r["file"],allow_pickle=False) as z:
                    f=z["rho_gram_factor"]
                    self.assertAlmostEqual(float(np.vdot(f,f).real),1,places=10)
            for b in range(4):
                self.assertFalse((pathlib.Path(td)/f"KATO_BUFFER_{b}.npz").exists())
    def test_missing_evidence_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                reconstruct(pathlib.Path(td)/"out",pathlib.Path(td)/"missing")
if __name__=="__main__":unittest.main()
