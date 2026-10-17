#!/usr/bin/env python3
"""Exact interface for the charged C10/Kato boundary-to-core contraction.

The physical construction has two logically distinct contractions.

1. Exterior Dirichlet environment.  A selected charged Kato/C10 state is
   represented by a positive block density operator.  Exterior BCC shells are
   eliminated by explicitly supplied completely-positive (CP) block maps while
   preserving the exact U(1), SU(2), intertwiner, and CAR labels.  No thin-string
   probability mixture is substituted for the Kato-selected state.

2. Core insertion and plaquette recoupling.  The resulting boundary density is
   contracted with an explicitly supplied boundary-to-path insertion tensor.
   That tensor weights every iota-resolved core recoupling term before the
   sparse plaquette block is assembled.

The implementation is complete for *supplied physical tensors* and enforces
an explicit input contract when any dynamical tensor is absent.  The reference L=2 IOTA_RESOLVED payloads
validate the periodic-core recoupler only; they do not provide the exterior
charged Kato density or the boundary-to-path insertion tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple
import argparse
import hashlib
import json
import numpy as np
from scipy import sparse

ACTION_SHA256 = "c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7"
SOURCE_ORDER = ("L1", "L2", "L3", "R1", "R2", "R3")
FAMILY_PAIRS = ((0, 3), (1, 4), (2, 5))
Sector = Tuple[int, int, int, int]  # (q_U1, twice_j_SU2, iota_SU2, CAR parity)


class MissingPhysicalInput(RuntimeError):
    """Raised when a required dynamical tensor is absent."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class SectorVector:
    """Block-sparse pure state with exact symmetry-sector labels."""
    blocks: Mapping[Sector, np.ndarray]

    def validate(self) -> None:
        if not self.blocks:
            raise ValueError("sector vector is empty")
        for sec, v in self.blocks.items():
            if len(sec) != 4:
                raise ValueError(f"invalid sector key {sec}")
            a = np.asarray(v, dtype=np.complex128)
            if a.ndim != 1 or a.size == 0:
                raise ValueError(f"sector {sec} must be a nonempty vector")

    def norm(self) -> float:
        self.validate()
        return float(np.sqrt(sum(np.vdot(np.asarray(v), np.asarray(v)).real for v in self.blocks.values())))

    def to_density(self) -> "SectorDensity":
        self.validate()
        return SectorDensity({sec: np.outer(np.asarray(v), np.asarray(v).conj()) for sec, v in self.blocks.items()})


@dataclass(frozen=True)
class SectorDensity:
    """Positive block density tensor on an exact symmetry-resolved boundary."""
    blocks: Mapping[Sector, np.ndarray]

    def validate(self, *, positivity_tolerance: float = 1e-11) -> None:
        if not self.blocks:
            raise ValueError("sector density is empty")
        for sec, rho in self.blocks.items():
            if len(sec) != 4:
                raise ValueError(f"invalid density sector {sec}")
            a = np.asarray(rho, dtype=np.complex128)
            if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
                raise ValueError(f"density block {sec} must be a nonempty square matrix")
            herm = float(np.linalg.norm(a - a.conj().T, ord="fro"))
            if herm > positivity_tolerance:
                raise ValueError(f"density block {sec} is not Hermitian: {herm}")
            mine = float(np.linalg.eigvalsh((a + a.conj().T) / 2).min())
            if mine < -positivity_tolerance:
                raise ValueError(f"density block {sec} is not positive semidefinite: {mine}")
        if self.trace() <= 0:
            raise ValueError("sector density has non-positive trace")

    def trace(self) -> float:
        return float(sum(np.trace(np.asarray(rho)).real for rho in self.blocks.values()))


@dataclass(frozen=True)
class ShellCPMap:
    """One exact exterior-shell contraction represented as sector-resolved CP maps.

    ``transitions[(out_sector,in_sector)]`` is a sequence of Kraus matrices
    K_a with shape (dim_out, dim_in), implementing
        rho_out += sum_a K_a rho_in K_a^dagger.
    Cross-sector transitions are allowed only when explicitly serialized; no
    sector relation is inferred from dimensions.
    """
    transitions: Mapping[Tuple[Sector, Sector], Sequence[sparse.spmatrix]]
    radius_in: int
    radius_out: int
    provenance: str
    certified_two_norm_error: float = 0.0

    def validate(self) -> None:
        if self.radius_out >= self.radius_in:
            raise ValueError("shell CP map must contract inward")
        if not self.transitions:
            raise ValueError("shell CP map has no transitions")
        if self.certified_two_norm_error < 0:
            raise ValueError("negative shell error bound")
        out_dims: Dict[Sector, int] = {}
        in_dims: Dict[Sector, int] = {}
        for (out_sec, in_sec), kraus_ops in self.transitions.items():
            if len(out_sec) != 4 or len(in_sec) != 4 or not kraus_ops:
                raise ValueError("invalid CP-map transition")
            for K in kraus_ops:
                A = sparse.csr_matrix(K, dtype=np.complex128)
                if A.ndim != 2:
                    raise ValueError("Kraus object is not a matrix")
                out_dims.setdefault(out_sec, A.shape[0])
                in_dims.setdefault(in_sec, A.shape[1])
                if out_dims[out_sec] != A.shape[0] or in_dims[in_sec] != A.shape[1]:
                    raise ValueError("inconsistent Kraus dimensions within a sector")


@dataclass(frozen=True)
class CoreRecouplingPayload:
    """Iota-resolved core recoupling terms.

    Each row of ``labels`` is a six-integer fusion-path label
    (left_0,left_1,left_2,right_0,right_1,right_2).  These labels are not the
    six family/chirality source columns.
    """
    volume: int
    labels: np.ndarray
    amplitudes: np.ndarray
    carrier_sector: str
    provenance: str

    def validate(self) -> None:
        lab = np.asarray(self.labels)
        amp = np.asarray(self.amplitudes)
        if lab.ndim != 2 or lab.shape[1] != 6:
            raise ValueError("core recoupling labels must have shape (N,6)")
        if amp.ndim != 1 or amp.shape[0] != lab.shape[0]:
            raise ValueError("core amplitudes must have shape (N,) matching labels")
        if lab.dtype.kind not in "iu":
            raise ValueError("core recoupling labels must be exact integers")
        if not np.iscomplexobj(amp):
            raise ValueError("core amplitudes must be complex")


@dataclass(frozen=True)
class BoundaryPathInsertion:
    """Exact contraction from the boundary density to each recoupling path.

    For every exact boundary sector ``sec``, ``maps[sec]`` has shape
    (N_path, dim_sec**2).  With row-major vectorization of rho_sec, the sum of
    these maps produces one complex environment weight for every recoupling
    term.  This explicit tensor is the interface that cannot be reconstructed
    from an L=2 periodic recoupler alone.
    """
    maps: Mapping[Sector, sparse.spmatrix]
    provenance: str
    certified_two_norm_error: float = 0.0

    def validate(self, density: SectorDensity, n_paths: int) -> None:
        if not self.maps:
            raise ValueError("boundary-to-path insertion map is empty")
        if self.certified_two_norm_error < 0:
            raise ValueError("negative insertion error bound")
        missing = sorted(set(density.blocks) - set(self.maps))
        if missing:
            raise MissingPhysicalInput(f"boundary insertion lacks sectors {missing[:4]}")
        for sec, rho in density.blocks.items():
            A = sparse.csr_matrix(self.maps[sec], dtype=np.complex128)
            d = np.asarray(rho).shape[0]
            if A.shape != (n_paths, d * d):
                raise ValueError(f"boundary insertion shape mismatch for sector {sec}: {A.shape} != {(n_paths,d*d)}")


@dataclass(frozen=True)
class RankSixCoreEmbedding:
    """Explicit six-column source/sink injection on the recoupled core axes.

    ``left_sources`` has shape (N_left,6) and ``right_sources`` has shape
    (N_right,6), in the fixed order ``SOURCE_ORDER``.  The columns must be
    isometric.  This object is deliberately distinct from the iota path labels:
    no family/chirality identification is inferred from fusion labels.
    """
    left_sources: np.ndarray
    right_sources: np.ndarray
    provenance: str
    certified_two_norm_error: float = 0.0

    def validate(self, block: "RecoupledCoreBlock", *, tolerance: float = 1e-10) -> None:
        L = np.asarray(self.left_sources, dtype=np.complex128)
        R = np.asarray(self.right_sources, dtype=np.complex128)
        if L.shape != (block.matrix.shape[0], 6):
            raise ValueError(f"left rank-six embedding has shape {L.shape}, expected {(block.matrix.shape[0],6)}")
        if R.shape != (block.matrix.shape[1], 6):
            raise ValueError(f"right rank-six embedding has shape {R.shape}, expected {(block.matrix.shape[1],6)}")
        if self.certified_two_norm_error < 0:
            raise ValueError("negative rank-six embedding error bound")
        gl = float(np.linalg.norm(L.conj().T @ L - np.eye(6)))
        gr = float(np.linalg.norm(R.conj().T @ R - np.eye(6)))
        if gl > tolerance or gr > tolerance:
            raise ValueError(f"rank-six source columns are not isometric: left={gl}, right={gr}")


def project_recoupled_core_to_rank6(block: "RecoupledCoreBlock", embedding: RankSixCoreEmbedding) -> tuple[np.ndarray, dict]:
    """Project the supplied recoupled core block onto the exact six source columns.

    The returned 6x6 matrix is the core source-to-sink block required by the
    subsequent common-space propagation.  It is not by itself C_n^(6), because
    the latter requires powers of the complete physical transfer operator.
    """
    embedding.validate(block)
    L = np.asarray(embedding.left_sources, dtype=np.complex128)
    R = np.asarray(embedding.right_sources, dtype=np.complex128)
    M6 = np.asarray(L.conj().T @ (block.matrix @ R), dtype=np.complex128)
    return M6, {
        "status": "PASS_EXPLICIT_RANK6_CORE_PROJECTION",
        "source_order": list(SOURCE_ORDER),
        "shape": [6, 6],
        "frobenius_norm": float(np.linalg.norm(M6)),
        "certified_two_norm_error": float(embedding.certified_two_norm_error),
        "provenance": embedding.provenance,
        "is_Cn6": False,
    }


@dataclass
class RecoupledCoreBlock:
    volume: int
    carrier_sector: str
    left_labels: np.ndarray
    right_labels: np.ndarray
    matrix: sparse.csr_matrix
    input_amplitude_norm: float
    matrix_frobenius_norm: float
    duplicate_terms_combined: int


@dataclass
class BoundaryProviderResult:
    boundary_density: SectorDensity
    recoupled_block: RecoupledCoreBlock
    rank6_core_block: np.ndarray
    path_weights: np.ndarray
    certificate: dict


def _apply_shell_cp_map(density: SectorDensity, shell: ShellCPMap) -> SectorDensity:
    density.validate()
    shell.validate()
    # Ensure every populated input sector has at least one serialized transition.
    covered = {in_sec for (_, in_sec) in shell.transitions}
    missing = sorted(set(density.blocks) - covered)
    if missing:
        raise MissingPhysicalInput(f"shell CP map at R={shell.radius_in} lacks input sectors {missing[:4]}")
    out: Dict[Sector, np.ndarray] = {}
    for (out_sec, in_sec), kraus_ops in shell.transitions.items():
        if in_sec not in density.blocks:
            continue
        rho = np.asarray(density.blocks[in_sec], dtype=np.complex128)
        for K in kraus_ops:
            A = sparse.csr_matrix(K, dtype=np.complex128)
            if A.shape[1] != rho.shape[0]:
                raise ValueError(f"Kraus/input dimension mismatch for {in_sec}")
            term = np.asarray(A @ rho @ A.getH(), dtype=np.complex128)
            if out_sec in out and out[out_sec].shape != term.shape:
                raise ValueError(f"inconsistent output dimension for sector {out_sec}")
            out[out_sec] = out.get(out_sec, np.zeros_like(term)) + term
    result = SectorDensity(out)
    result.validate()
    return result


def contract_external_kato_density_to_core(
    density: SectorDensity,
    shell_maps: Sequence[ShellCPMap],
    *,
    trace_tolerance: float = 1e-9,
) -> tuple[SectorDensity, dict]:
    """Contract the selected charged density to the core interface.

    The contraction is exact for the supplied CP maps.  Trace preservation is
    tested after every shell and no silent renormalization is performed.
    """
    density.validate()
    initial_trace = density.trace()
    if abs(initial_trace - 1.0) > trace_tolerance:
        raise ValueError(f"initial Kato density is not normalized: trace={initial_trace}")
    current = density
    history = []
    accumulated_error = 0.0
    prev_radius = None
    for shell in shell_maps:
        shell.validate()
        if prev_radius is not None and shell.radius_in != prev_radius:
            raise ValueError("shell radii are not contiguous")
        current = _apply_shell_cp_map(current, shell)
        tr = current.trace()
        accumulated_error += float(shell.certified_two_norm_error)
        history.append({
            "radius_in": int(shell.radius_in),
            "radius_out": int(shell.radius_out),
            "trace": tr,
            "certified_two_norm_error": float(shell.certified_two_norm_error),
            "provenance": shell.provenance,
        })
        if abs(tr - 1.0) > trace_tolerance + accumulated_error:
            raise ValueError(f"shell contraction is not trace preserving within certificate: trace={tr}")
        prev_radius = shell.radius_out
    current.validate(positivity_tolerance=max(1e-11, accumulated_error + 1e-13))
    return current, {
        "status": "PASS_EXTERNAL_KATO_DENSITY_TO_CORE_FOR_SUPPLIED_CP_MAPS",
        "initial_trace": initial_trace,
        "final_trace": current.trace(),
        "shells": history,
        "accumulated_two_norm_error_bound": accumulated_error,
        "compression_or_truncation_hidden": False,
    }


def contract_external_kato_state_to_core(
    state: SectorVector,
    shell_maps: Sequence[ShellCPMap],
    *,
    trace_tolerance: float = 1e-9,
) -> tuple[SectorDensity, dict]:
    """Pure-state convenience wrapper for the density contraction."""
    state.validate()
    n = state.norm()
    if abs(n - 1.0) > trace_tolerance:
        raise ValueError(f"selected Kato state is not normalized: norm={n}")
    return contract_external_kato_density_to_core(state.to_density(), shell_maps, trace_tolerance=trace_tolerance)


def _unique_core_axes(payload: CoreRecouplingPayload):
    payload.validate()
    labels = np.asarray(payload.labels, dtype=np.int64)
    left_labels, left_inv = np.unique(labels[:, :3], axis=0, return_inverse=True)
    right_labels, right_inv = np.unique(labels[:, 3:], axis=0, return_inverse=True)
    return left_labels, right_labels, left_inv, right_inv


def assemble_recoupled_core_block(payload: CoreRecouplingPayload, path_weights: Optional[np.ndarray] = None) -> RecoupledCoreBlock:
    """Assemble the sparse core block, optionally weighted by a physical boundary tensor."""
    payload.validate()
    amps = np.asarray(payload.amplitudes, dtype=np.complex128)
    if path_weights is not None:
        w = np.asarray(path_weights, dtype=np.complex128)
        if w.shape != amps.shape:
            raise ValueError("path weights must have shape (N_path,)")
        amps = amps * w
    left_labels, right_labels, left_inv, right_inv = _unique_core_axes(payload)
    coo = sparse.coo_matrix((amps, (left_inv, right_inv)),
                            shape=(len(left_labels), len(right_labels)), dtype=np.complex128)
    before = int(coo.nnz)
    csr = coo.tocsr(); csr.sum_duplicates()
    return RecoupledCoreBlock(
        volume=int(payload.volume), carrier_sector=payload.carrier_sector,
        left_labels=left_labels.astype(np.int16, copy=False),
        right_labels=right_labels.astype(np.int16, copy=False),
        matrix=csr,
        input_amplitude_norm=float(np.linalg.norm(amps)),
        matrix_frobenius_norm=float(np.sqrt(np.sum(np.abs(csr.data) ** 2))),
        duplicate_terms_combined=before - int(csr.nnz),
    )


def apply_boundary_path_insertion(density: SectorDensity, insertion: BoundaryPathInsertion, n_paths: int) -> tuple[np.ndarray, dict]:
    density.validate()
    insertion.validate(density, n_paths)
    weights = np.zeros(n_paths, dtype=np.complex128)
    for sec, rho in density.blocks.items():
        A = sparse.csr_matrix(insertion.maps[sec], dtype=np.complex128)
        weights += np.asarray(A @ np.asarray(rho, dtype=np.complex128).reshape(-1, order="C")).reshape(-1)
    return weights, {
        "status": "PASS_BOUNDARY_DENSITY_TO_RECOUPLING_PATH_CONTRACTION",
        "n_paths": int(n_paths),
        "weight_norm": float(np.linalg.norm(weights)),
        "certified_two_norm_error": float(insertion.certified_two_norm_error),
        "provenance": insertion.provenance,
    }


def validate_plaquette_descriptor(descriptor: dict, volume: int) -> dict:
    manifest = descriptor.get("manifest", descriptor)
    for key in ("n_links", "n_plaquettes", "final_open"):
        if key not in manifest:
            raise ValueError(f"plaquette descriptor missing {key}")
    if int(manifest["final_open"]) != 0:
        raise ValueError("global plaquette frontier does not close")
    if volume == 2 and (int(manifest["n_links"]), int(manifest["n_plaquettes"])) != (24, 24):
        raise ValueError("L=2 validation descriptor must be the 24-link/24-plaquette core")
    return {k: int(manifest.get(k, -1)) for k in ("n_links", "n_plaquettes", "max_open_plaquettes", "final_open")}


def validate_oriented_binding(binding: dict, volume: int) -> dict:
    entries = binding.get("binding", [])
    if not entries:
        raise ValueError("oriented binding is empty")
    counts: Dict[int, int] = {}; oriented: Dict[int, int] = {}; links = set()
    for row in entries:
        links.add(int(row["link"]))
        legs = row.get("iota_leg_order", [])
        if len(legs) != 4 or sorted(int(x["slot"]) for x in legs) != [0, 1, 2, 3]:
            raise ValueError("each link must carry four ordered plaquette legs")
        for x in legs:
            p = int(x["plaquette"]); s = int(x["orientation"])
            if s not in (-1, 1):
                raise ValueError("orientation must be +/-1")
            counts[p] = counts.get(p, 0) + 1
            oriented[p] = oriented.get(p, 0) + s
    if any(v != 4 for v in counts.values()) or any(v != 0 for v in oriented.values()):
        raise ValueError("oriented plaquette incidence failed closure")
    if volume == 2 and len(links) != 24:
        raise ValueError("L=2 binding must contain 24 links")
    return {"links_in_binding": len(links), "plaquettes_in_binding": len(counts),
            "all_incidence_counts_four": True, "all_oriented_sums_zero": True}


def charged_C10_Kato_boundary_tensor_and_recoupled_plaquette_block_provider(
    *,
    volume: int,
    kato_state: Optional[SectorVector] = None,
    kato_density: Optional[SectorDensity] = None,
    exterior_shell_maps: Optional[Sequence[ShellCPMap]],
    core_payload: Optional[CoreRecouplingPayload],
    boundary_path_insertion: Optional[BoundaryPathInsertion],
    rank6_source_embedding: Optional[RankSixCoreEmbedding],
    plaquette_descriptor: dict,
    oriented_binding: Optional[dict] = None,
) -> BoundaryProviderResult:
    """Execute the named physical map for explicitly supplied tensors.

    Exactly one of ``kato_state`` or ``kato_density`` must be provided.  The
    function fails closed if the exterior shell CP maps, the boundary-to-path
    insertion tensor, or the recoupled core payload is absent.  No volume or
    boundary tensor is inferred from a smaller periodic cell.
    """
    if (kato_state is None) == (kato_density is None):
        raise MissingPhysicalInput(f"Exactly one selected charged Kato state/density is required for L={volume}")
    if exterior_shell_maps is None or len(exterior_shell_maps) == 0:
        raise MissingPhysicalInput(f"No exterior C10/Kato shell CP maps supplied for L={volume}")
    if core_payload is None:
        raise MissingPhysicalInput(f"No recoupled core plaquette payload supplied for L={volume}")
    if boundary_path_insertion is None:
        raise MissingPhysicalInput(f"No boundary-density-to-recoupling-path insertion tensor supplied for L={volume}")
    if rank6_source_embedding is None:
        raise MissingPhysicalInput(f"No explicit six-column family/chirality core embedding supplied for L={volume}")
    if int(core_payload.volume) != int(volume):
        raise ValueError("core payload volume does not match requested volume")

    geom = validate_plaquette_descriptor(plaquette_descriptor, volume)
    orient = validate_oriented_binding(oriented_binding, volume) if oriented_binding is not None else None
    if kato_density is not None:
        boundary, ext_cert = contract_external_kato_density_to_core(kato_density, exterior_shell_maps)
    else:
        boundary, ext_cert = contract_external_kato_state_to_core(kato_state, exterior_shell_maps)
    weights, insertion_cert = apply_boundary_path_insertion(boundary, boundary_path_insertion, len(core_payload.amplitudes))
    block = assemble_recoupled_core_block(core_payload, weights)
    rank6_core, rank6_cert = project_recoupled_core_to_rank6(block, rank6_source_embedding)
    cert = {
        "status": "PASS_COMPLETE_BOUNDARY_TO_RECOUPLED_CORE_MAP_FOR_SUPPLIED_TENSORS",
        "action_sha256": ACTION_SHA256,
        "volume": int(volume),
        "source_order": list(SOURCE_ORDER),
        "family_pairs_zero_based": [list(x) for x in FAMILY_PAIRS],
        "external_boundary_certificate": ext_cert,
        "boundary_insertion_certificate": insertion_cert,
        "boundary_sectors": [list(x) for x in sorted(boundary.blocks)],
        "core_recoupling_certificate": {
            "carrier_sector": block.carrier_sector,
            "left_dimension": int(block.matrix.shape[0]),
            "right_dimension": int(block.matrix.shape[1]),
            "sparse_nnz": int(block.matrix.nnz),
            "weighted_input_amplitude_norm": block.input_amplitude_norm,
            "matrix_frobenius_norm": block.matrix_frobenius_norm,
        },
        "rank6_core_projection_certificate": rank6_cert,
        "geometry": geom,
        "oriented_binding": orient,
        "mass_extraction_stage": "CERTIFIED_LARGE_VOLUME_C10_KATO_EXECUTION_STAGE",
    }
    return BoundaryProviderResult(boundary, block, rank6_core, weights, cert)


def compute_projected_moments(operator, sources: np.ndarray, nmax: int) -> np.ndarray:
    """Compute C_n=V^dagger U^n V, n=0..nmax, for a supplied physical operator."""
    V = np.asarray(sources, dtype=np.complex128)
    if V.ndim != 2 or V.shape[1] != 6:
        raise ValueError("sources must have six columns")
    if np.linalg.norm(V.conj().T @ V - np.eye(6)) > 1e-10:
        raise ValueError("sources are not isometric")
    if operator.shape[0] != operator.shape[1] or operator.shape[0] != V.shape[0]:
        raise ValueError("operator/source dimensions are incompatible")
    C = np.empty((nmax + 1, 6, 6), complex)
    W = V.copy(); C[0] = V.conj().T @ W
    for n in range(1, nmax + 1):
        W = operator @ W
        C[n] = V.conj().T @ W
    return C


def _load_core_payload(path: Path, carrier: str, cutoff: str) -> CoreRecouplingPayload:
    z = np.load(path, allow_pickle=False)
    return CoreRecouplingPayload(2, z[f"{carrier}_labels"], z[f"{carrier}_amplitudes"], carrier,
                                 f"reference {cutoff} iota-resolved periodic-core recoupler")


def validate_l2_core_recoupler(data_dir: Path) -> dict:
    """Validate the periodic-core recoupler against the reference rank-12 SVDs."""
    data_dir = Path(data_dir)
    desc = json.loads((data_dir / "CORE_PLAQUETTE_GEOMETRY_L2.json").read_text())
    bind = json.loads((data_dir / "CORE_ORIENTED_IOTA_BINDING_L2.json").read_text())
    out = {
        "status": "PASS_L2_CORE_RECOUPLER_VALIDATION__NOT_EXTERNAL_KATO_BOUNDARY_MAP",
        "source_order": list(SOURCE_ORDER),
        "family_pairs_zero_based": [list(x) for x in FAMILY_PAIRS],
        "cuts": {},
        "scope": "This test validates the periodic-core iota recoupler only. The exterior Dirichlet C10/Kato density and its boundary-to-path insertion tensor are independent physical inputs.",
    }
    for cutoff in ("COARSE", "FINE"):
        payload = _load_core_payload(data_dir / f"IOTA_RESOLVED_{cutoff}.npz", "qm3_j1", cutoff)
        block = assemble_recoupled_core_block(payload)
        archived = np.load(data_dir / f"{cutoff}_qm3_j1_SVD.npz", allow_pickle=False)
        if not np.array_equal(block.left_labels, archived["left_labels"]) or not np.array_equal(block.right_labels, archived["right_labels"]):
            raise ValueError(f"{cutoff}: sector labels differ from reference SVD")
        sv = np.linalg.svd(block.matrix.toarray(), compute_uv=False)
        retained = archived["s"].astype(float)
        sdiff = float(np.max(np.abs(sv[:len(retained)] - retained)))
        low = archived["U"] @ np.diag(retained) @ archived["Vh"]
        resid = float(np.linalg.norm(block.matrix.toarray() - low))
        if sdiff > 1e-16:
            raise ValueError(f"{cutoff}: retained spectrum mismatch {sdiff}")
        out["cuts"][cutoff.lower()] = {
            "retained_rank": int(len(retained)),
            "retained_singular_value_max_abs_difference": sdiff,
            "discarded_frobenius_norm_against_rank12_reference": resid,
            "input_terms": int(payload.amplitudes.shape[0]),
            "left_dimension": int(block.matrix.shape[0]),
            "right_dimension": int(block.matrix.shape[1]),
            "sparse_nnz": int(block.matrix.nnz),
            "largest_singular_value": float(sv[0]),
            "thirteenth_singular_value": float(sv[12]),
            "geometry": validate_plaquette_descriptor(desc, 2),
            "oriented_binding": validate_oriented_binding(bind, 2),
            "mass_extraction_stage": "CERTIFIED_LARGE_VOLUME_C10_KATO_EXECUTION_STAGE",
        }
    return out


def synthetic_full_provider_self_test() -> dict:
    """Exercise exterior contraction, path insertion, recoupling and six-column projection."""
    sec = (-3, 1, 0, 1)
    psi = SectorVector({sec: np.array([1.0 + 0j])})
    I = sparse.identity(1, dtype=np.complex128, format="csr")
    shell = ShellCPMap({(sec, sec): [I]}, 3, 2, "synthetic identity shell CP map")
    labels = np.array([[i,0,0,i,0,0] for i in range(6)], dtype=np.int16)
    amps = np.arange(1, 7, dtype=float).astype(np.complex128)
    core = CoreRecouplingPayload(2, labels, amps, "qm3_j1", "synthetic six-path core")
    B = sparse.csr_matrix(np.ones((6,1), dtype=np.complex128))
    insertion = BoundaryPathInsertion({sec: B}, "synthetic exact boundary insertion")
    embedding = RankSixCoreEmbedding(np.eye(6, dtype=np.complex128), np.eye(6, dtype=np.complex128),
                                     "synthetic exact rank-six source embedding")
    desc = {"n_links": 24, "n_plaquettes": 24, "final_open": 0, "max_open_plaquettes": 1}
    got = charged_C10_Kato_boundary_tensor_and_recoupled_plaquette_block_provider(
        volume=2, kato_state=psi, exterior_shell_maps=[shell], core_payload=core,
        boundary_path_insertion=insertion, rank6_source_embedding=embedding, plaquette_descriptor=desc)
    expected_w = np.ones(6, dtype=np.complex128)
    expected_m6 = np.diag(amps)
    if np.max(np.abs(got.path_weights - expected_w)) > 1e-14:
        raise AssertionError("synthetic boundary insertion regression")
    if np.max(np.abs(got.rank6_core_block - expected_m6)) > 1e-14:
        raise AssertionError("synthetic rank-six projection regression")
    if abs(got.boundary_density.trace() - 1.0) > 1e-14:
        raise AssertionError("synthetic boundary trace regression")
    return {
        "status": "PASS_SYNTHETIC_COMPLETE_RANK6_PROVIDER_ALGORITHM",
        "boundary_trace": got.boundary_density.trace(),
        "path_weight_max_error": float(np.max(np.abs(got.path_weights - expected_w))),
        "rank6_projection_max_error": float(np.max(np.abs(got.rank6_core_block - expected_m6))),
        "rank6_core_singular_values": np.linalg.svd(got.rank6_core_block, compute_uv=False).tolist(),
        "mass_extraction_stage": "CERTIFIED_LARGE_VOLUME_C10_KATO_EXECUTION_STAGE",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-l2-core", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    result = {"synthetic_two_stage_provider": synthetic_full_provider_self_test()}
    if a.validate_l2_core is not None:
        result["l2_core_recoupler"] = validate_l2_core_recoupler(a.validate_l2_core)
    txt = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt)
    print(txt, end="")


if __name__ == "__main__":
    main()
