import tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hpc_gate import validate,REQUIRED
class HPCGateTest(unittest.TestCase):
    def test_missing_all_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            result=validate(td)
            self.assertFalse(result['HPC_PRODUCTION_AUTHORIZED'])
            self.assertEqual(result['status'],'BLOCKED_MISSING_INPUT')
            self.assertEqual(len(result['errors']),len(REQUIRED))
    def test_empty_cert_files_refuse(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            (d/(REQUIRED[0]+'.json')).write_text('{}')
            result=validate(td)
            self.assertEqual(result['status'],'BLOCKED_MISSING_INPUT')
            self.assertFalse(result['HPC_PRODUCTION_AUTHORIZED'])
if __name__=='__main__':unittest.main()
