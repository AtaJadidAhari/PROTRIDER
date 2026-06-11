# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PROTRIDER detects protein abundance outliers in mass spectrometry-based proteomics data. It trains a conditional autoencoder to reconstruct expected protein intensities, then flags sample–protein pairs whose residuals (observed − expected) fall in the tails of a fitted distribution. Published in Bioinformatics (doi:10.1093/bioinformatics/btaf628).

## Development commands

The project uses `uv` (see CI in `.github/workflows/tests.yml`):

```bash
uv sync --all-groups            # install runtime + dev deps
uv run pytest tests/ -q         # run the full test suite
uv run pytest tests/test_pipeline_features.py -v          # one module
uv run pytest tests/test_pipeline_misc.py::test_name      # one test
uv run pytest tests/ --cov=protrider                      # with coverage

# End-to-end smoke test on the bundled sample data
uv run protrider run --config config.yaml
uv run protrider plot --config config.yaml all
```

Tests are grouped into `Test*` classes and rely on fixtures in `tests/conftest.py` that point at `sample_data/`. Most tests pin `find_q_method` to a small integer to skip latent-dimension search and stay deterministic. Note `tests/README.md` references a `test_pipeline_cv.py` module that is not in the repo; treat its module list as approximate.

## Architecture

The pipeline is a linear flow orchestrated by `pipeline.run(config)` ([src/protrider/pipeline.py](src/protrider/pipeline.py)):

1. **Config** — `ProtriderConfig` ([src/protrider/config.py](src/protrider/config.py)) is a dataclass that validates parameters and derives computed fields in `__post_init__`: `log_func`/`base_fn` from `log_func_name`, and `device_torch` from `device` + CUDA availability. The CLI loads it via `load_config()`; the Python API constructs it directly. Computed fields are excluded from `save()`/`as_dict()`.

2. **Dataset** — `ProtriderDataset` ([src/protrider/datasets/datasets.py](src/protrider/datasets/datasets.py)) reads intensities, applies DESeq2 size-factor normalization (skippable via `normalize=False`) + log transform, filters proteins exceeding `max_allowed_NAs_per_protein`, and parses covariates. **Internal orientation is samples × proteins** (rows × columns) regardless of the input file's orientation (`input_format`). NaNs are masked: the model sees mean-imputed values (`X`), but loss, residuals, and p-values are computed only over observed entries via `torch_mask`. It mixes in `PCADataset` for SVD used by both latent-dim selection and PCA weight initialization.

3. **Latent dimension** — `find_latent_dim()` ([src/protrider/model/model_helper.py](src/protrider/model/model_helper.py)) picks the encoding dim `q` via `OHT` (optimal hard threshold on singular values, fast default), `gs`/`bs` (grid/binary search that injects synthetic outliers and maximizes AUPRC of recovering them), or a fixed integer.

4. **Model** — `ProtriderAutoencoder` ([src/protrider/model/model.py](src/protrider/model/model.py)) is a conditional autoencoder. Covariates are concatenated to inputs at both encoder and decoder (`cond`). With `n_layers=1` it is effectively linear and can be initialized from PCA (`initialize_wPCA`). `presence_absence=True` adds a second BCE head predicting missingness (only validated for `n_layers=1`); loss is `MSEBCELoss` = masked MSE + `lambda_bce` · BCE.

5. **Statistics** — [src/protrider/stats.py](src/protrider/stats.py) fits a per-protein distribution to residuals (`gaussian`, or `t` via a two-pass fit that shares a common degrees-of-freedom for stability), computes two-sided and left-sided p-values + z-scores, and adjusts with BH/BY (`adjust_pvals`). Heavy column-wise fits are parallelized with joblib (`n_jobs`).

6. **Results** — `run()` returns `(Result, ModelInfo, FitParameters, GridSearchResult)`. `Result` ([src/protrider/pipeline.py](src/protrider/pipeline.py)) holds all output DataFrames and handles `save(format="wide"|"long")` and plotting. Long format (`protrider_summary.csv`) is the headline output and filters to outliers (`PROTEIN_PADJ <= outlier_threshold`) unless `report_all`.

### Checkpointing
`run()` saves the trained model to `checkpoint_path` (or `<out_dir>/model.pt`). If that file already exists it is **loaded and training is skipped** — delete it or change the path to force retraining. The checkpoint stores `q`, `n_layers`, and `presence_absence` so the architecture is reconstructed on load.

## Entry points

- **CLI** ([src/protrider/cli.py](src/protrider/cli.py)): `protrider run` and the chained `protrider plot` group, both driven by a YAML config. Plot commands read previously-saved CSVs from `out_dir`.
- **Python API** ([src/protrider/__init__.py](src/protrider/__init__.py)): exposes `run`, `ProtriderConfig`, `load_config`, `Result`, `ModelInfo`. `Result`/`ModelInfo` plot methods return plotnine objects when `out_dir` is omitted.

## Conventions

- Pandas convention throughout the internals: **rows = samples (`sampleID`), columns = proteins (`proteinID`)**; CSVs are transposed back to proteins × samples on save in wide format.
- Torch tensors are `double` precision and moved to `device_torch`.
- `find_q_method` is a string even for integers (validated in `ProtriderConfig.__post_init__`).
- Optional `wandb` logging is gated behind `use_wandb` and imported lazily.
