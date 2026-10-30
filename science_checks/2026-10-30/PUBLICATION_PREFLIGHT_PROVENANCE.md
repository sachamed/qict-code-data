# QICT publication-preflight data lineage — 30 October 2026

The corrected research release is `QICT_Publication_Preflight_Corrected_2026-10-30.zip` (SHA-256 `005ab03f61213c5bbfa44cccbb1cd6cf3d2989e64d801add095cdf484ab284b5`). The corrected package itself is **not uploaded in this pull request**.

The scientific-file manifest originally disagreed with exactly two included files:
- `reproducibility/code/regenerate_reference_figures.py`
- `reproducibility/results/reference_figures/VISUAL_SPEC.json`

The source and visual specification name exactly the same ten figures. After updating those two manifest hashes and adding the immutable origin ledger, the released package verifier checks **1156/1156** entries.

All 31 vector figure assets match their registered SHA-256 hashes. Seven original generators or raw inputs remain unidentified. These **archival vector figures** are:

| Figure file | Original recorded SHA-256 |
|---|---|
| `neutral_Mnu_heatmap.pdf` | `a26b4a7c983dc0fbfce4bcbdc44f4efecac689473882340a5ea44f33bbef36e9` |
| `endpoint_action_Z3.pdf` | `fe8379add018054f5f07c8c125339ca43c14be372c147c8bdc06273225c2e600` |
| `fig2_operator_bounds.pdf` | `2bd7837a8741e26d7119cf360124cb731d26caaaac819b1dcb54d62fa74e87a8` |
| `correlation_realization.pdf` | `bba17e7d137a36d9cc6b9e2d5193254053ea73e737f4f34667627c9486813c95` |
| `figS1_dressing_control.pdf` | `198552ad561af25a30789d4686b61bdec82621d09d6ee69fb7b401ddfcc5c643` |
| `figS2_retained_transfer.pdf` | `f0a5abd7d988a142663438155e67a88479f2b4887eb078de1da140c4de259081` |
| `c8j_free_limit_phases.pdf` | `85a11ec58a784db25f1bf174d59ccd541bfe8e1e434aa49607de4110c43ebea6` |

Both accompanying PDFs were recompiled (120 and 760 pages). All 235 named companion-document targets were checked and no link destination is missing. Their shared bibliography contains 156 entries with a numbered Supplemental Material citation. Eight finite checks passed, including 17 C10/GNS unit tests and 2,081-state neutral SU(2) operator checks.

**This is a data-lineage record, not a claim that the physical charged C8/C10 amplitudes, unique limiting vacuum, masses, CKM/PMNS, or graviton response have been calculated.** Binary matrices, historical datasets and complete independent figure producers have not been added or replaced by this PR.
