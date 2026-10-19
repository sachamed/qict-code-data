import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contract_inspection import inspect,ROOT
class ContractTests(unittest.TestCase):
    def test_frozen_action_source_is_not_executable_operator(self):
        a=inspect(ROOT/'evidence/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json')
        self.assertEqual(a['status'],'BLOCKED_MISSING_INPUT')
        self.assertFalse(a['contract']['interaction_kernel_apply_U'])
        self.assertFalse(a['contract']['four_interacting_lambda1_state_densities'])
        self.assertTrue(a['contract']['U1_SU2_SU3_generator_formulae'])
if __name__=='__main__': unittest.main()
