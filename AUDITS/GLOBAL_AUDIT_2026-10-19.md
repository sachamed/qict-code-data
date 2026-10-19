# QICT code and data — global corrective audit

**Assessment date:** 19 October 2026  
**Repository:** https://github.com/sachamed/qict-code-data  
**Inspected baseline:** `f4583aad21615a5b8dddd71010fd6767bf1d5ec6`  
**Scientific status:** finite numerical calculations and source preservation verified in the tested scope; the fully interacting physical mass reconstruction remains `BLOCKED_MISSING_INPUT`.

## Confirmed findings and corrections

| ID | Finding | Severity | Correction |
|---|---|---|---|
| QA-01 | CI wrote a fresh SHA-256 file and immediately checked that same file; this is a self-consistency check, not a check against a frozen source manifest. | High | Added `tools/verify_scientific_release.py`, comparing each of the 137 source release files with the independently published SHA-256 and Git blob digests. CI now runs that verifier, while the generated self-consistency list remains explicitly labelled as such. |
| QA-02 | The CI workflow did not run tests or compile all 47 scripts in the October 19 scientific releases. | High | Added release-tree Python compilation, the primitive mathematical test suite (66 tests), and dark-sector finite operator checks to CI. |
| QA-03 | HPC input screening did not consistently reject non-finite numerical input before linear algebra and sometimes exposed lower-level LAPACK errors instead of a clear diagnostic refusal. | Moderate | Added explicit finite, dimension, tolerance and numerical conditioning guards to `source_isometry`, `check_density`, `block_toeplitz`, and `moment_diagnostic`. |
| QA-04 | Absence of negative controls for release-file tampering, missing/extra files, invalid manifest paths, symbolic links and duplicate manifest entries. | Moderate | Added an independent verification test suite with nine negative/positive cases; separate numerical refusal tests cover 11 cases. |

## Checks performed

- Audited Git repository structure at the inspected baseline (1,070 Git tree entries). This structural inventory is **not** a statement that each program was dynamically executed.
- Byte-level verification of **137/137** scientific files from the two source archives against the published release manifest: **PASS**.
- AST parse of all **47** released Python scripts: **PASS**.
- Original Haar/Z6 tests: **66/66 PASS** on Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0.
- Original targeted dark-sector operator closure and finite matrix certificate checks: **PASS**.
- Release integrity verifier: **9/9 PASS** (including tampering tests).
- Numerical data guard tests: **11/11 PASS**; original HPC guard tests: **5/5 PASS**.
- PDF integrity: **18/18 PDF files readable** by `pdfinfo` and `pdftotext`; 73 pages total for the released documents and figures (44 main, 7 supplement, 7 predictions, remaining 15 figure files). This is **not** an editorial proofread of all PDF mathematical assertions.
- All 15 released NPZ archives were opened with `numpy.load(..., allow_pickle=False)` and every entry read: **PASS**.
- The entire cosmological full-PDF LaTeX recompilation and a fresh HPC cluster run were **not** performed in this audit. Historical source packaging and the author's frozen PDFs remain unchanged.

## Known scientific limitations retained explicitly

- The provided Kato buffers, quantum C10 shell maps, rank-six carrier, full interacting U-cone provider and physical six-channel moments have **not** been certified as one closed physical computational chain.
- Finite primitive resolvent calculations and conditional reference cosmology cannot establish a QICT physical dark-matter mass, relic density or cosmological constant without further operator-level derivations.
- The new software checks screen mathematical consistency and data provenance, but do not establish continuum limits or physical interpretations.
- No computational audit can guarantee the absence of every latent defect; this report lists the checks actually executed and the unverified portions.

## Reproduction

```bash
python tools/verify_scientific_release.py
python -m unittest discover -s tests -v
python -m unittest discover -s QICT_FULL_PHYSICAL_EXECUTION/tests -v
cd QICT_RESEARCH_RELEASES/2026-10-19/PRIMITIVE_HAAR_Z6/QICT_PRIMITIVE_INFINITY_CERTIFIED_2026_10_19
python -m unittest discover -s tests -v
```

From the dark-sector reproducibility directory:

```bash
python code/validate_operator_certificate.py
python code/test_operator_closure.py
```

**Release policy:** the 137 original scientific release files, their SHA-256 manifest, original numerical results and frozen articles are preserved unchanged. Corrections are made to audit tools, preflight validation, unit tests, CI and documentation, not retroactively to research datasets.
## Conservative cleanup addendum — 19 October 2026

The current repository was checked for generated build artifacts, editor backups and cache objects; none were found among tracked files. The current tree contained 50 identical-content blob groups (51 additional paths), most inside frozen source snapshots or required HPC/data compatibility layouts. Those provenance-relevant paths were retained. Two strictly redundant paths were selected for deletion:

1. `MANIFEST_SHA256.txt` — 288 hash records, all equivalent to `GITHUB_EXPORT_SHA256.txt` after normalizing each entry's leading `./` prefix; the preserved latter file is already documented as a historical export.
2. `QICT_FULL_PHYSICAL_EXECUTION/hpc/run_l24_ucone.slurm` — exact Git object duplicate of retained `QICT_FULL_PHYSICAL_EXECUTION/hpc/run_l24.slurm`; neither HPC operational data nor scientific routines are removed.

No records from the 137-file October releases, nor any Kato, C9/C10, G2, matrices, tensors, figures, the October 17 snapshot, frozen action, tests, or certificates, are deleted. Git object identity and immutable prior commits preserve the deleted path history. No claim is made that all identical-path occurrences are obsolete or that full physical QICT calculation is completed.
