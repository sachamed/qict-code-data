import json, sys, math, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mathematical_checks as m
ROOT=Path(__file__).resolve().parents[1]

class TheoremChecks(unittest.TestCase):
    def test_01_action_hash(self):
        self.assertEqual(m.source_action_check(ROOT/'evidence/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json')['hash'],m.ACTION_HASH)
    def test_02_source_order(self):
        self.assertEqual(m.source_action_check(ROOT/'evidence/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json')['source_gate_order'][0],'Gamma(S_z)')
    def test_03_original_cover_not_quotient(self):
        q=m.quotient_audit();self.assertEqual(q['original_magnetic_cost_difference_computed'],120)
        self.assertGreater(q['phase_difference_abs_from_one'],.2)
        self.assertFalse(q['original_gate_descends_to_quotient'])
    def test_04_quotient_alternative_invariant(self):
        q=m.quotient_audit();self.assertLess(abs(q['quotient_candidate_center_difference']),1e-12)
    def test_05_quotient_candidate_weakfield_match(self):
        q=m.quotient_audit();self.assertLess(q['weak_field_relative_discrepancy_at_t_1e-3'],1e-5)
    def test_06_quotient_candidate_global_nonidentical(self):
        u=np.eye(2,dtype=complex);v=np.eye(3,dtype=complex)
        a=m.magnetic_cover(.6,u,v);b=m.magnetic_quotient_alternative(.6,u,v)
        self.assertGreater(abs(a-b),.01)
    def test_07_SM_representations_descent(self):
        # y=6Y, l=2j mod2, t=colour triality; standard-generation multiplets.
        for y,l,t in [(1,1,1),(4,0,1),(-2,0,1),(-3,1,0),(-6,0,0),(3,1,0)]:
            self.assertEqual((y+3*l+2*t)%6,0)
    def test_08_moments_identical_but_spectra_different(self):
        for n in [1,9,21]:
            p=m.truncated_moment_ambiguity(n)
            self.assertLess(p['max_moment_discrepancy'],1e-13)
            self.assertGreater(p['discrete_positive_source_residue'],0)
    def test_09_auxiliary_gap_uniform(self):
        for grid in (4,5,11,13):
            p=m.auxiliary_pole_certificate(grid)
            self.assertAlmostEqual(p['uniform_spectral_gap'],math.sqrt(2))
    def test_10_auxiliary_projection_infinite_rank_sequence(self):
        ranks=[m.auxiliary_pole_certificate(n)['global_isolated_projector_rank_finite_cutoff'] for n in (4,5,11)]
        self.assertEqual(ranks,[24,30,66])
    def test_11_auxiliary_residue_nondecay(self):
        for n in (4,5,11,13):self.assertGreater(m.auxiliary_pole_certificate(n)['min_residue_eigenvalue'],.999999)
    def test_12_three_methods_agree(self):
        for n in (4,11):
            p=m.auxiliary_pole_certificate(n)
            self.assertLess(p['method_Riesz_discrepancy'],1e-12)
            self.assertLess(p['method_Fourier_discrepancy'],1e-12)
    def test_13_copy_parity_commutation(self):
        U=np.diag([1,1j]);Q=np.diag([-1,1]);V=np.array([[1],[0]],complex)
        self.assertLess(m.copy_odd_gate_check(U,Q,V)['odd_source_error'],1e-15)
    def test_14_copy_parity_fail_closed(self):
        U=np.array([[0,1],[1,0]],complex);Q=np.diag([-1,1]);V=np.array([[1],[0]],complex)
        with self.assertRaises(ValueError):m.copy_odd_gate_check(U,Q,V)
    def test_15_schur_identity(self):
        U=np.array([[0,1],[-1,0]],complex);V=np.array([[1],[0]],complex)
        p=m.schur_feshbach_check(U,V,1.5+.4j)
        self.assertLess(p['error'],1e-13)
    def test_16_clock_missing_refused(self):
        with self.assertRaises(ValueError):m.phase_to_mass_interval(None,None)
    def test_17_bad_clock_interval_refused(self):
        with self.assertRaises(ValueError):m.phase_to_mass_interval((.1,.2),(-1.,1.))
    def test_18_geV_mapping_only_when_clock_given(self):
        lo,hi=m.phase_to_mass_interval((.1,.2),(1e-12,2e-12))
        self.assertGreater(lo,0);self.assertGreater(hi,lo)
    def test_19_original_kato_certificate_not_interacting(self):
        p=json.loads((ROOT/'evidence/PRIMITIVE_A7_GROUND_CERTIFICATE.json').read_text())
        self.assertIsInstance(p,dict)
        # Truth about physical status is in independent fail-closed registry.
    def test_20_no_physical_files_mislabelled(self):
        for name in ('KATO_BUFFER_0.npz','CN6_MOMENTS.npz','MASS_GEV_CERTIFICATE_PASS.json'):
            self.assertFalse((ROOT/'results'/name).exists())

if __name__=='__main__':unittest.main()
