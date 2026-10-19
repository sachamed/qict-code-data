"""Recompute the supplied C9 moments from their finite operators and seeds.

The stationary construction follows compute_finite_C9_common_state.py in the
author-supplied QICT_C9_STATIONARY_SCIENTIFIC_SUBMISSION_2026-10-15 archive.
Joint neutral eigenvalue clusters retain their coherences. This is a floating
point reproduction check; certify_operator_spectra.py supplies the separate
outward-rounded operator certificate.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import schur

ROOT = Path(__file__).resolve().parents[1]


def cluster_indices(z, tolerance=1e-12):
    groups = []
    for index in np.argsort(np.angle(z)):
        if not groups or abs(z[index]-z[groups[-1][0]]) > tolerance:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    if len(groups) > 1 and abs(z[groups[0][0]]-z[groups[-1][0]]) <= tolerance:
        groups[0] = groups[-1]+groups[0]
        groups.pop()
    return [np.asarray(g, dtype=int) for g in groups]


def coherence_bound(coefficients, groups):
    # Sum each off-diagonal rank-one matrix's trace norm; no cancellation.
    result = 0.
    for ids in groups:
        for i, j in zip(*np.triu_indices(len(ids), 1)):
            result += 2.*abs(coefficients[ids[i]])*abs(coefficients[ids[j]])
    return float(result)


def main():
    data = np.load(ROOT/'code/FINITE_C9_COMMON_STATE_AND_MOMENTS.npz',
                   allow_pickle=False)
    records = []
    for r in range(1, 5):
        def a(name):
            return data[f'R{r}_{name}']
        TY, QY = schur(a('neutral_U1'), output='complex')
        TW, QW = schur(a('neutral_SU2'), output='complex')
        ly, lw = np.diag(TY), np.diag(TW)
        lam = (ly[:, None]*lw[None, :]).ravel()
        c = np.kron(QY.conj().T@a('primitive_U1'),
                    QW.conj().T@a('primitive_SU2'))
        groups = cluster_indices(lam)
        count = len(lw)
        pY = [np.eye(len(ly), dtype=complex) for _ in range(2)]
        pW = np.eye(a('charged_SU2').shape[0], dtype=complex)
        O = a('weak_source')
        moments = []
        for n in range(13):
            by = [QY.conj().T@p@QY for p in pY]
            bw = QW.conj().T@O.conj().T@pW@O@QW
            sector_values = []
            for sector in (0, 1):
                value = 0j
                for ids in groups:
                    iy, iw = ids//count, ids % count
                    bs = by[sector][iy[:, None], iy[None, :]]
                    if sector == 0:
                        bs = bs*bw[iw[:, None], iw[None, :]]
                    else:
                        bs = bs*(iw[:, None] == iw[None, :])*(lw[iw][:, None]**n)
                    value += np.vdot(c[ids], bs@(c[ids]*lam[ids]**(-n)))
                sector_values.append(value)
            moments.append(np.diag([sector_values[0]]*3+[sector_values[1]]*3))
            pY = [a('left_U1')@pY[0], a('right_U1')@pY[1]]
            pW = a('charged_SU2')@pW
        moments = np.asarray(moments)
        error = float(np.max(abs(moments-a('moments'))))
        assert error < 2e-12
        original_groups = [np.flatnonzero(a('joint_cluster_ids') == k)
                           for k in np.unique(a('joint_cluster_ids'))]
        row = {'regulator': f'R{r}', 'moment_orders': [0, 12],
               'max_entry_discrepancy': error,
               'joint_cluster_count': len(groups),
               'joint_dephasing_coherence_trace_norm_majorant':
                   coherence_bound(a('primitive_coefficients'), original_groups),
               'computed_basis_coherence_trace_norm_majorant': coherence_bound(c, groups)}
        records.append(row)
        print(json.dumps(row), flush=True)
    (ROOT/'data/C9_OPERATOR_RECONSTRUCTION.json').write_text(json.dumps({
        'scope': 'Floating-point reproduction of the recorded joint-dephasing construction',
        'joint_cluster_chord_tolerance': 1e-12, 'regulators': records,
    }, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
