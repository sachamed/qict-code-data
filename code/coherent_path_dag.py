"""Coherent operator-valued path dressing on a finite directed acyclic graph.

Implements the existing QICT prescription D_v = sum_w p_vw U_vw D_w,
D_sink = I, without enumerating paths. Callbacks act on a common Hilbert
space and may accept a vector or a block of source vectors.

This is an operator contraction algorithm, not a physical C10 provider.
It does not construct missing microscopic gates, select a Kato state, or
supply independently certified operator/roundoff error bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class Edge:
    source: Hashable
    target: Hashable
    probability: float
    apply: Callable[[np.ndarray], np.ndarray]
    apply_adjoint: Callable[[np.ndarray], np.ndarray]
    # Supplied bounds refer to the applied approximation of a true contraction.
    operator_error_bound: float = 0.0
    approximate_norm_bound: float = 1.0


class CoherentPathDAG:
    def __init__(self, order: Sequence[Hashable], sink: Hashable,
                 edges: Sequence[Edge], normalization_tolerance: float = 1e-12):
        self.order = tuple(order)
        self.sink = sink
        self.edges = tuple(edges)
        if (not math.isfinite(normalization_tolerance)
                or normalization_tolerance < 0):
            raise ValueError("normalization tolerance must be finite and nonnegative")
        positions = {v: i for i, v in enumerate(self.order)}
        if not self.order or len(positions) != len(self.order) or sink not in positions:
            raise ValueError("provide a nonempty topological order with unique nodes and sink")
        self.outgoing = {v: [] for v in self.order}
        for index, edge in enumerate(self.edges):
            if edge.source not in positions or edge.target not in positions:
                raise ValueError("edge endpoint absent from topological order")
            if positions[edge.source] >= positions[edge.target]:
                raise ValueError("edge violates the supplied acyclic order")
            for value in (edge.probability, edge.operator_error_bound,
                          edge.approximate_norm_bound):
                if not math.isfinite(value) or value < 0:
                    raise ValueError("weights and error/norm bounds must be finite and nonnegative")
            if not callable(edge.apply) or not callable(edge.apply_adjoint):
                raise ValueError("each edge needs operator and adjoint callbacks")
            self.outgoing[edge.source].append((index, edge))
        if self.outgoing[sink]:
            raise ValueError("sink must have no outgoing edges")
        residuals = []
        for v in self.order:
            if v == sink:
                continue
            row_sum = math.fsum(e.probability for _, e in self.outgoing[v])
            residuals.append(abs(row_sum - 1.0))
            if not self.outgoing[v] or abs(row_sum - 1.0) > normalization_tolerance:
                raise ValueError("each nonsink node must have normalized outgoing weights")
        self.normalization_residual = max(residuals, default=0.0)

    @staticmethod
    def _array(value, shape=None):
        a = np.asarray(value, dtype=np.complex128)
        if a.ndim not in (1, 2) or a.size == 0 or not np.isfinite(a).all():
            raise ValueError("operator arguments/results must be finite nonempty vectors or blocks")
        if shape is not None and a.shape != shape:
            raise ValueError("edge callback changed the common Hilbert-space shape")
        return a

    def _check_source(self, source):
        if source not in self.outgoing:
            raise ValueError("source absent from graph")

    def apply(self, source, vector):
        """Apply D_source by backward recursion; O(number of edges) callbacks."""
        self._check_source(source)
        x = self._array(vector)
        values = {self.sink: x.copy()}
        for v in reversed(self.order):
            if v == self.sink:
                continue
            result = np.zeros_like(x)
            for _, edge in self.outgoing[v]:
                if edge.probability:
                    result += edge.probability * self._array(
                        edge.apply(values[edge.target].copy()), x.shape)
            values[v] = result
        return values[source]

    def apply_adjoint(self, source, vector):
        """Apply D_source dagger by forward propagation of accumulated messages."""
        self._check_source(source)
        x = self._array(vector)
        messages = {source: x.copy()}
        for v in self.order:
            if v not in messages or v == self.sink:
                continue
            for _, edge in self.outgoing[v]:
                if not edge.probability:
                    continue
                term = edge.probability * self._array(
                    edge.apply_adjoint(messages[v].copy()), x.shape)
                if edge.target in messages:
                    messages[edge.target] += term
                else:
                    messages[edge.target] = term.copy()
        return messages.get(self.sink, np.zeros_like(x))

    def flow_weights(self, source):
        """Expected edge occupations for the declared path weights."""
        self._check_source(source)
        mass = {v: 0.0 for v in self.order}
        mass[source] = 1.0
        weights = np.zeros(len(self.edges), dtype=float)
        for v in self.order:
            for index, edge in self.outgoing[v]:
                weights[index] = mass[v] * edge.probability
                mass[edge.target] += weights[index]
        return weights

    def error_diagnostic(self, source):
        """Evaluate a conditional analytic bound; no floating-point certification.

        For true edge contractions U and approximations A with certified
        ||A-U|| <= epsilon and ||A|| <= s, use
            b_v = sum p_vw b_w,
            a_v = sum p_vw (epsilon_vw b_w + s_vw a_w),
        with b_sink=1 and a_sink=0. Then ||D_v-Dtilde_v|| <= a_v.
        For exactly normalized weights and s<=1 this reduces to
        sum_e flow_weight[e] * epsilon[e]. This routine does not include
        probability uncertainty, callback arithmetic, or rounding in the sum.
        """
        self._check_source(source)
        true_norm = {self.sink: 1.0}
        approximate_norm = {self.sink: 1.0}
        error = {self.sink: 0.0}
        for v in reversed(self.order):
            if v == self.sink:
                continue
            true_norm[v] = math.fsum(
                e.probability * true_norm[e.target] for _, e in self.outgoing[v])
            approximate_norm[v] = math.fsum(
                e.probability * e.approximate_norm_bound * approximate_norm[e.target]
                for _, e in self.outgoing[v])
            error[v] = math.fsum(
                e.probability * (e.operator_error_bound * true_norm[e.target]
                    + e.approximate_norm_bound * error[e.target])
                for _, e in self.outgoing[v])
        return {
            "classification": "CONDITIONAL_ERROR_BOUND_FLOAT_EVALUATION",
            "normalization_residual": self.normalization_residual,
            "true_dressing_norm_bound": true_norm[source],
            "approximate_dressing_norm_bound": approximate_norm[source],
            "operator_error_bound_excluding_roundoff": error[source],
            "expected_path_length": math.fsum(self.flow_weights(source)),
            "physical_C10_certified": False,
            "roundoff_certified": False,
        }
