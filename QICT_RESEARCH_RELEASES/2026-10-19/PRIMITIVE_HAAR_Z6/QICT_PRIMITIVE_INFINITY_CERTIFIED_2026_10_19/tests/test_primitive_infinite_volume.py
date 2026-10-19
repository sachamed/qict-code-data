import json
import sys
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from primitive_infinite_volume_certificate import (run,primitive_exact_certificate,
    exact_pairwise_density_distance_upper,file_digest,sqrt_upper_decimal)
from central_character_rule import (z6_character_residue,is_quotient_representation,
                                    build_report,mixed_character,central_step)

class PrimitiveInfiniteVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=run()

    def test_energy_intervals_intersect(self):
        rows=self.report['rows']
        self.assertEqual([x['support_flux_N'] for x in rows],[8,12,16,20])
        lo=max(Decimal(r['ground_energy_lower_rational_outward']) for r in rows)
        hi=min(Decimal(r['ground_energy_upper_rational_outward']) for r in rows)
        self.assertLessEqual(lo,hi)
        self.assertGreater(lo,Decimal('3.0897227917851729'))
        self.assertLess(hi,Decimal('3.0897227917851730'))

    def test_exact_primitive_certificates_prove_overlap(self):
        for r in self.report['rows']:
            self.assertTrue(all(r['exact_checks'].values()))
            self.assertLess(Decimal(r['trace_distance_to_infinite_ground_upper_exact_comparison']),Decimal('1e-13'))
            self.assertEqual(r['physical_interacting_KATO_BUFFER'],False)

    def test_pairwise_exact_common_embedding(self):
        self.assertEqual(len(self.report['pairwise_bounds']),6)
        for r in self.report['pairwise_bounds']:
            self.assertLessEqual(Decimal(r['trace_distance_direct_dyadic_upper']),
                                 Decimal(r['trace_distance_pairwise_upper_from_infinite_ground']))

    def test_bad_flux_and_zero_rejected(self):
        for flux,psi in [(np.array([-1,1]),np.array([1.,2.])),
                         (np.arange(-1,2),np.zeros(3))]:
            with self.assertRaises(ValueError):primitive_exact_certificate(flux,psi)

    def test_complex_dyadic_rejected_in_real_scope(self):
        with self.assertRaises(ValueError):
            primitive_exact_certificate(np.arange(-1,2),np.array([0,1j,0]))

    def test_exact_upper_rounding(self):
        b=sqrt_upper_decimal(Fraction(2,9),18)
        self.assertGreater(Decimal(b),Decimal('0.471404520791'))
        self.assertLess(Decimal(b),Decimal('0.471404520792'))

    def test_source_provenance(self):
        import json
        c=json.loads((ROOT/'evidence/PRIMITIVE_A7_GROUND_CERTIFICATE.json').read_text())
        for r in c['cutoffs']:
            self.assertEqual(file_digest(ROOT/'evidence'/r['file']),r['sha256'])

class CentralCharacterTests(unittest.TestCase):
    def test_representation_selection_rule(self):
        self.assertTrue(is_quotient_representation(1,1,1))
        self.assertTrue(is_quotient_representation(6,0,0))
        self.assertTrue(is_quotient_representation(0,2,0))
        self.assertFalse(is_quotient_representation(0,1,0))
        self.assertFalse(is_quotient_representation(0,0,1))
        self.assertFalse(is_quotient_representation(1,0,0))
        self.assertTrue(is_quotient_representation(-5,1,1))
        self.assertEqual(z6_character_residue(1,1,1),0)

    def test_invalid_labels_rejected(self):
        for x in [(0,-1,0),(0,1,3),(0,1,.5)]:
            with self.assertRaises(ValueError):z6_character_residue(*x)

    def test_mixed_character_central_invariance(self):
        rep=build_report()
        self.assertLess(rep['six_central_lifts_max_character_error_float64'],1e-13)
        self.assertFalse(rep['frozen_action_modified'])

if __name__=='__main__':unittest.main()

class PrimitiveLocalPoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from primitive_pole_triple_method import create_pole_certificate
        cls.c=create_pole_certificate()

    def test_genuine_primitive_resolvent_atom_with_certified_positive_residue(self):
        c=self.c
        zmin=Decimal(c['spectral_residue_at_site_zero_lower_exact_rational_outward'])
        zmax=Decimal(c['spectral_residue_at_site_zero_upper_exact_rational_outward'])
        self.assertGreater(zmin,Decimal('0.9058'))
        self.assertLess(zmax,Decimal('0.906'))
        self.assertLess(zmax-zmin,Decimal('5e-17'))
        self.assertTrue(c['rigorously_isolated_hamiltonian_ground_pole'])
        self.assertFalse(c['physical_QICT_charged_or_dark_pole'])

    def test_energy_interval_and_gap(self):
        c=self.c
        a=Decimal(c['ground_energy_lower_exact_rational_outward'])
        b=Decimal(c['ground_energy_upper_exact_rational_outward'])
        self.assertLess(a,b)
        self.assertLess(b-a,Decimal('1e-26'))
        self.assertGreater(a,Decimal('3.08'))
        self.assertEqual(c['minimum_contour_spectral_distance_lower'],2)

    def test_all_three_algorithms_converge(self):
        rows=self.c['three_calculations']['cutoffs']
        self.assertGreaterEqual(len(rows),4)
        for row in rows:
            self.assertLess(row['pole_methods_difference'],1e-12)
            self.assertLess(row['residue_methods_difference'],1e-12)

    def test_schur_tail_contour_derivative(self):
        from primitive_pole_triple_method import schur_tail_and_derivative,schur_equation
        z=2.7
        n=6
        f,df=schur_tail_and_derivative(z,n)
        h=1e-6
        num=(schur_tail_and_derivative(z+h,n)[0]-schur_tail_and_derivative(z-h,n)[0])/(2*h)
        self.assertLess(abs(df-num),2e-9)
        self.assertAlmostEqual(schur_equation(z,n),z-4-8*f,places=12)

class PrimitiveSplitStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from primitive_splitstep_diagnostic import run
        cls.report=run()

    def test_unwrapped_H_ground_is_not_splitstep_eigenvector(self):
        rows=self.report['numerical_optimized_phase_defect']
        self.assertEqual(len(rows),4)
        for r in rows:
            self.assertGreater(r['optimized_Floquet_eigenphase_defect'],0.03)
            self.assertLess(r['optimized_Floquet_eigenphase_defect'],0.04)
            self.assertLess(r['independent_padded_expm_overlap_discrepancy'],2e-13)
        self.assertFalse(self.report['physical_Floquet_pole_certified'])

    def test_unity_operator_defect_formula(self):
        from primitive_splitstep_diagnostic import defect_from_overlap
        self.assertEqual(defect_from_overlap(1+0j),0.)
        self.assertAlmostEqual(defect_from_overlap(0.+0j),np.sqrt(2),places=14)

    def test_two_algorithms_are_genuinely_distinct(self):
        from primitive_splitstep_diagnostic import overlap_bessel,overlap_padded_matrix
        v=np.array([0.3,0.8,0.2],dtype=complex)
        flux=np.array([-1,0,1])
        self.assertLess(abs(overlap_bessel(v,flux)-overlap_padded_matrix(v,flux)),1e-13)

class PhysicalScopeStatusTests(unittest.TestCase):
    def test_primitive_pole_cannot_unlock_dark_mass_or_hpc(self):
        from subprocess import run
        p=run([sys.executable,str(ROOT/'src/generate_status.py')],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        status=json.loads((ROOT/'results/PHYSICAL_CLOSURE_STATUS.json').read_text())
        self.assertFalse(status['all_13_requirements_closed'])
        self.assertFalse(status['HPC_production_permitted'])
        self.assertIsNone(status['physical_mass_GeV'])
        self.assertIsNone(status['physical_pole'])
        self.assertEqual(status['further_certified_mathematics']['primitive_local_residue_lower_analytic'],'1/2')
        self.assertFalse(status['further_certified_mathematics']['full_interacting_dark_physics_promoted'])

class PrimitiveFloquetShiftTheoremTests(unittest.TestCase):
    def test_exact_phase_translation_symmetry(self):
        from primitive_floquet_shift_no_go import exact_phase_periodicity
        self.assertTrue(all(exact_phase_periodicity(r) for r in range(-500,501)))
        self.assertTrue(all(exact_phase_periodicity(r,shift=63) for r in range(-500,501)))
        self.assertFalse(exact_phase_periodicity(1,shift=21))

    def test_rank_finite_Floquet_forbidden_only_in_subsystem(self):
        from primitive_floquet_shift_no_go import theorem
        a=theorem()
        self.assertTrue(a['all_nonzero_finite_rank_Riesz_projections_of_U_BE_forbidden'])
        self.assertTrue(a['primitive_H_rank_one_ground_projector_allowed'])
        self.assertFalse(a['full_QICT_micro_operator_no_go_proved'])
