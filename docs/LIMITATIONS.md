# LIMITATIONS.md

Known limitations and caveats that apply to the project as a whole,
independent of any single phase. This is not a place for phase-specific
open items (those live in `DATA_AUDIT.md` / `DECISIONS.md` / `EXPERIMENTS.md`)
— only for caveats that can't be resolved by better code, because they are
properties of the released dataset itself.

---

## L1 — Unknown provenance of the released feature standardization

Every one of the 165 feature columns in `elliptic_txs_features.csv` is
already globally standardized (mean ≈ 0, std ≈ 1 per column) by the dataset
authors before public release — verified directly in `docs/DATA_AUDIT.md`
section 4.

We do not know, and cannot verify from the released files alone, whether the
mean/std used for that standardization were computed only from an early
subset of time steps or from all 49 time steps (including what are, from our
chronological split's perspective, "future" val/test time steps).

**What this does and does not mean:**

- If the standardization statistics were computed over all 49 time steps,
  then a small amount of dataset-wide aggregate information (the global
  mean/std per feature) is baked into every feature value we receive, before
  our pipeline ever touches the data. This would technically mean the
  released features are not *strictly* independent of the val/test period.
- This is **not** the same as label leakage, and it does **not** mean our
  split, training, or evaluation code leaks information — every check in
  `src/data/preprocessing.py` (`verify_no_leakage`) and every test in
  `tests/test_data.py` confirms the split itself is correctly chronological
  and that no label or index information crosses from val/test into train.
- We have **not** confirmed this actually occurred, and we are not claiming
  it did. It is a documented, unresolved possibility about the public
  release's preprocessing, not a finding.
- It is also not something this project can correct without access to the
  original, unnormalized feature values, which were never released.

**How to treat this:** when reporting results, any claim of a "fully
leakage-free pipeline" should be qualified with this caveat — our pipeline
introduces no leakage, but we cannot rule out a small amount already present
in the upstream released data. This is a known, general caveat about the
public Elliptic release, not something specific to this codebase.
