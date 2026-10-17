#!/usr/bin/env python3
"""QICT C7 spacetime-lift verification.

This script uses the specified finite transfer matrices and the parameter-free
3D Weyl-QCA formulas adopted by C7.  It does not read particle-data tables.
It verifies:
  * tetrahedral/BCC second-moment isotropy;
  * unitarity of both Weyl branches;
  * unitarity of the matrix-Julia Dirac lift for u,d,e,nu_D;
  * equality of zero-momentum rest quasienergy magnitudes with the
    principal Julia angles arcsin(singular_values(T_f)).
"""
from pathlib import Path
import csv, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CERT = ROOT / "certificates"; CERT.mkdir(exist_ok=True)

# ---------- specified finite transfer matrices ----------
T = {}
with (DATA / "family_transfer_matrices.csv").open() as f:
    for row in csv.DictReader(f):
        sec = row["sector"]
        T.setdefault(sec, np.zeros((3,3), dtype=complex))
        T[sec][int(row["i"]), int(row["j"])] = float(row["real"]) + 1j*float(row["imag"])

# ---------- BCC tetrahedral generator geometry ----------
h = np.array([
    [ 1, 1, 1],
    [ 1,-1,-1],
    [-1, 1,-1],
    [-1,-1, 1],
], dtype=float) / math.sqrt(3.0)
first_moment = h.sum(axis=0)
second_moment = sum(np.outer(v,v) for v in h)
isotropy_target = (4.0/3.0)*np.eye(3)

# ---------- two parameter-free 3D Weyl-QCA branches ----------
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]], complex)
sz = np.array([[1,0],[0,-1]], complex)
PAULI = (sx,sy,sz)

def A_weyl(k, branch="-"):
    kx,ky,kz = map(float,k)
    cx,cy,cz = [math.cos(q/math.sqrt(3.0)) for q in (kx,ky,kz)]
    sxv,syv,szv = [math.sin(q/math.sqrt(3.0)) for q in (kx,ky,kz)]
    if branch == "+":
        d = cx*cy*cz + sxv*syv*szv
        n = np.array([
            sxv*cy*cz - cx*syv*szv,
            -cx*syv*cz - sxv*cy*szv,
            cx*cy*szv - sxv*syv*cz,
        ])
    elif branch == "-":
        d = cx*cy*cz - sxv*syv*szv
        n = np.array([
            sxv*cy*cz + cx*syv*szv,
            cx*syv*cz - sxv*cy*szv,
            cx*cy*szv + sxv*syv*cz,
        ])
    else:
        raise ValueError(branch)
    return d*np.eye(2, dtype=complex) - 1j*sum(n[i]*PAULI[i] for i in range(3))

def sqrt_psd(A):
    A = (A + A.conj().T)/2
    w,V = np.linalg.eigh(A)
    if w.min() < -1e-11:
        raise RuntimeError(f"not PSD: min eigenvalue {w.min()}")
    w = np.clip(w,0,None)
    return (V*np.sqrt(w)) @ V.conj().T

def dirac_lift(Tf,k,branch="-"):
    """Matrix-Julia generalization of the 3D Dirac QCA.

    U_f(k) = [[A(k) D_L, i T], [i T^dag, A(k)^dag D_R]],
    with tensor products over the 2-component spacetime Weyl spinor.
    D_L=sqrt(I-TT^dag), D_R=sqrt(I-T^dag T).
    """
    A = A_weyl(k,branch)
    I2 = np.eye(2,dtype=complex)
    I3 = np.eye(3,dtype=complex)
    DL = sqrt_psd(I3 - Tf@Tf.conj().T)
    DR = sqrt_psd(I3 - Tf.conj().T@Tf)
    return np.block([
        [np.kron(A,DL),             1j*np.kron(I2,Tf)],
        [1j*np.kron(I2,Tf.conj().T), np.kron(A.conj().T,DR)],
    ])

# deterministic test grid, no random seed dependence
grid = [
    (0.0,0.0,0.0),
    (0.1,0.2,-0.3),
    (0.4,-0.7,1.1),
    (1.0,1.0,1.0),
    (2.0,1.0,0.5),
    (-1.7,0.8,2.2),
]
weyl_res = {}
for br in ("+","-"):
    weyl_res[br] = max(float(np.linalg.norm(A_weyl(k,br).conj().T@A_weyl(k,br)-np.eye(2))) for k in grid)

sector = {}
for sec,Tf in T.items():
    svals = np.sort(np.linalg.svd(Tf,compute_uv=False))
    angles = np.arcsin(svals)
    max_u = 0.0
    for br in ("+","-"):
        for k in grid:
            U=dirac_lift(Tf,k,br)
            max_u=max(max_u,float(np.linalg.norm(U.conj().T@U-np.eye(U.shape[0]))))
    U0=dirac_lift(Tf,(0,0,0),"-")
    phases=np.sort(np.abs(np.angle(np.linalg.eigvals(U0))))
    # each principal rest angle occurs four times (2 spin x +/- phase magnitude)
    expected=np.sort(np.repeat(angles,4))
    phase_match=float(np.max(np.abs(phases-expected)))
    sector[sec] = {
        "transfer_singular_values": [float(x) for x in svals],
        "principal_rest_angles": [float(x) for x in angles],
        "max_unitarity_residual_on_test_grid": max_u,
        "zero_momentum_phase_match_residual": phase_match,
    }

cert = {
    "theory":"QICT C7 spacetime lift",
    "external_data_opened": False,
    "bcc_geometry": {
        "tetrahedral_generators": h.tolist(),
        "first_moment_norm": float(np.linalg.norm(first_moment)),
        "second_moment_isotropy_residual": float(np.linalg.norm(second_moment-isotropy_target)),
        "translation_rank": 3,
        "weyl_spinor_dimension": 2,
        "weyl_branch_count": 2,
        "max_weyl_unitarity_residual": weyl_res,
    },
    "matrix_julia_dirac_lift": sector,
    "interpretation": {
        "rest_angles": "bare C6 finite-generator quasienergies, not renormalized physical masses",
        "physical_observables": "must be extracted from exact interacting CTP/1PI poles and proper vertices",
    },
    "status":"PASS" if max(weyl_res.values()) < 1e-12 and all(v["max_unitarity_residual_on_test_grid"]<1e-11 and v["zero_momentum_phase_match_residual"]<1e-11 for v in sector.values()) else "FAIL"
}
(CERT/"c7_dirac_lift_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")

with (DATA/"c7_rest_quasienergy_angles.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["sector","branch","principal_angle_rad","status"])
    for sec,v in sector.items():
        for i,a in enumerate(v["principal_rest_angles"],1):
            w.writerow([sec,i,repr(a),"BARE_C6_EQUALS_C7_REST_QUASIENERGY"])

print(json.dumps(cert,indent=2))
