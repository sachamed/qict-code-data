"""Independent finite checks for coherent paths and the SVD budget regression."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from coherent_path_dag import CoherentPathDAG, Edge

LOCAL = ROOT / "snapshot-2026-10-17/task_evidence/TASK1/code/microscopic_closure/rank6_tensor_local_blocks.py"
spec = importlib.util.spec_from_file_location("qict_local_blocks_regression", LOCAL)
local_blocks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = local_blocks
spec.loader.exec_module(local_blocks)


def rx(t):
    return np.array([[np.cos(t), 1j*np.sin(t)],
                     [1j*np.sin(t), np.cos(t)]], complex)


def ry(t):
    return np.array([[np.cos(t), np.sin(t)],
                     [-np.sin(t), np.cos(t)]], complex)


def rz(t):
    return np.diag([np.exp(1j*t), np.exp(-1j*t)])


def edge(v, w, p, matrix, error=0.0, norm=1.0):
    a = np.asarray(matrix, complex)
    return Edge(v, w, p, lambda x: a @ x, lambda x: a.conj().T @ x, error, norm)


def fixture():
    return [(0, 1, .37, rx(.4)), (0, 2, .63, rz(.7)),
            (1, 2, .25, ry(-.2)), (1, 3, .75, rz(-.6)),
            (2, 3, 1.0, rx(1.1))]


def enumerate_paths(rows, v=0):
    if v == 3:
        return [(1.0, np.eye(2, dtype=complex))]
    return [(p*q, a @ b) for s, t, p, a in rows if s == v
            for q, b in enumerate_paths(rows, t)]


class CoherentPathTests(unittest.TestCase):
    def test_noncommuting_path_sum(self):
        rows = fixture()
        dag = CoherentPathDAG(range(4), 3, [edge(*r) for r in rows])
        explicit = sum(p*a for p, a in enumerate_paths(rows))
        computed = dag.apply(0, np.eye(2))
        np.testing.assert_allclose(computed, explicit, atol=3e-15, rtol=0)
        self.assertLessEqual(np.linalg.norm(computed, 2), 1 + 3e-15)

    def test_adjoint_and_block_sources(self):
        dag = CoherentPathDAG(range(4), 3, [edge(*r) for r in fixture()])
        matrix = dag.apply(0, np.eye(2))
        x = np.array([[.3+.4j, .2], [-.2+.5j, -.7j]], complex)
        np.testing.assert_allclose(dag.apply_adjoint(0, x),
                                   matrix.conj().T @ x, atol=3e-15, rtol=0)
        np.testing.assert_allclose(dag.apply(0, x[:, 0]),
                                   matrix @ x[:, 0], atol=3e-15, rtol=0)

    def test_gauge_covariance(self):
        rows = fixture()
        g = [ry(.31), rz(-.27), rx(.93), rx(.45) @ ry(-.3)]
        dag = CoherentPathDAG(range(4), 3, [edge(*r) for r in rows])
        transformed = CoherentPathDAG(range(4), 3, [
            edge(v, w, p, g[v] @ a @ g[w].conj().T) for v, w, p, a in rows])
        np.testing.assert_allclose(transformed.apply(0, np.eye(2)),
            g[0] @ dag.apply(0, np.eye(2)) @ g[3].conj().T, atol=3e-15, rtol=0)

    def test_destructive_interference_is_retained(self):
        identity = np.eye(2, dtype=complex)
        dag = CoherentPathDAG(range(4), 3, [
            edge(0, 1, .5, identity), edge(0, 2, .5, -identity),
            edge(1, 3, 1, identity), edge(2, 3, 1, identity)])
        np.testing.assert_allclose(dag.apply(0, identity), 0, atol=0, rtol=0)

    def test_flow_weighted_error(self):
        rows = fixture()
        eps = [.01, .02, .03, .04, .05]
        exact = CoherentPathDAG(range(4), 3, [edge(*r) for r in rows])
        approx = CoherentPathDAG(range(4), 3, [
            edge(v, w, p, (1-e)*a, e, 1-e)
            for (v, w, p, a), e in zip(rows, eps)])
        occupation = exact.flow_weights(0)
        np.testing.assert_allclose(occupation,
            [.37, .63, .0925, .2775, .7225], atol=1e-15, rtol=0)
        measured = np.linalg.norm(exact.apply(0, np.eye(2))
                                  - approx.apply(0, np.eye(2)), 2)
        report = approx.error_diagnostic(0)
        bound = report["operator_error_bound_excluding_roundoff"]
        self.assertLessEqual(measured, bound + 3e-15)
        self.assertLessEqual(bound, float(occupation @ eps) + 3e-15)
        self.assertAlmostEqual(report["expected_path_length"], 2.0925)
        self.assertFalse(report["physical_C10_certified"])
        self.assertFalse(report["roundoff_certified"])

    def test_invalid_graph_rejected(self):
        identity = np.eye(2)
        with self.assertRaises(ValueError):
            CoherentPathDAG([0, 1], 1, [edge(0, 1, .9, identity)])
        with self.assertRaises(ValueError):
            CoherentPathDAG([0, 1], 1, [edge(1, 0, 1, identity)])
        with self.assertRaises(ValueError):
            CoherentPathDAG([0, 0, 1], 1, [edge(0, 1, 1, identity)])
        with self.assertRaises(ValueError):
            CoherentPathDAG([0, 1], 1, [edge(0, 1, -1, identity)])

    def test_shape_change_rejected(self):
        dag = CoherentPathDAG([0, 1], 1, [
            Edge(0, 1, 1.0, lambda x: np.ones(3), lambda x: np.ones(3))])
        with self.assertRaises(ValueError):
            dag.apply(0, np.ones(2))


class TruncationBudgetTests(unittest.TestCase):
    def test_absolute_budget_cannot_be_overridden(self):
        matrix = np.diag([1.0, .11, .08])
        factors, report = local_blocks.block_svd(
            {("q",): matrix}, rel_tol=.2, abs_budget=.1)
        u, s, vh = factors[("q",)]
        self.assertEqual(len(s), 2)
        self.assertLessEqual(report["discarded_singular_value_sum_bound"], .1)
        self.assertLessEqual(np.linalg.norm(matrix-(u*s) @ vh, 2), .1)

    def test_all_local_self_checks(self):
        self.assertEqual(local_blocks.self_test()["status"], "PASS_LOCAL_BLOCKS")


if __name__ == "__main__":
    unittest.main()
