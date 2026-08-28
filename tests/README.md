# PROTRIDER Test Suite

The test suite is organized into multiple modules for better maintainability:

## Test Modules

### `test_pipeline_standard.py` (5 tests)
Tests for standard (non-CV) pipeline execution:
- File path inputs
- DataFrame inputs  
- No covariates
- Wide format output
- Long format output

### `test_pipeline_cv.py` (4 tests)
Tests for cross-validation modes:
- K-fold cross-validation
- Leave-one-out cross-validation (LOOCV)
- Early stopping in CV
- Fit every fold option

### `test_pipeline_config.py` (4 tests)
Tests for configuration validation:
- Missing input validation
- Invalid latent dimension method
- Negative epochs
- Invalid NA threshold

### `test_pipeline_features.py` (14 tests)
Tests for advanced configuration options:
- Log transformations (log, log2, log10, none)
- P-value distributions (gaussian, t)
- P-value adjustment methods (bh, by)
- Latent dimension methods (fixed, OHT, grid search)
- NA thresholds
- Batch size and learning rates
- PCA initializationww
- Outlier thresholds
- Presence/absence modeling

### `test_pipeline_misc.py` (10 tests)
Tests for output consistency and edge cases:
- P-value range validation
- Fold change consistency
- Name preservation (samples, proteins)
- Single/multiple covariates
- Custom pseudocount values
- Seed handling and reproducibility
- Configuration save/load
- Report options

## FRASER Tests (`FRASER/`)

Tests validating the FRASER (splicing outlier) port against R reference outputs:

### `test_data.py` (6 tests)
- K/N counts vs. R reference
- Expression filter vs. R reference
- Final filter count vs. R reference
- Logit transform vs. R reference
- Calculate counts matches K and N
- Jaccard index equals K over N

### `test_annotations.py` (1 test)
- Junction/gene annotation matches R reference

### `test_model.py` (5 tests)
- PCA reconstruction matches R predicted means
- Dispersion fit correlates with R rho
- Optimal SVHT coefficient matches R reference
- Median Marchenko-Pastur matches R reference
- OHT latent dimension finder recovers known rank

### `test_stats.py` (6 tests, 3 skipped)
- Beta-binomial p-values match R reference
- Holm p-value adjustment matches R reference
- Gene-level p-value shape matches number of genes
- *(skipped)* Beta-binomial p-value randomization — placeholder, nothing to test yet
- *(skipped)* Padj with rho cutoff — `adjust_pvals` has no `rhoCutoff` parameter yet
- *(skipped)* Padj on gene subset — no per-sample gene-subset FDR correction implemented yet

### `test_pipeline.py` (2 tests)
- End-to-end run with file path inputs
- Save results in long format

### `test_r_python_top_outliers.py` (2 tests)
End-to-end comparison against a real `fraser_run_timed.R` run (not just checked-in reference
CSVs), ranking junction/sample pairs purely by raw p-value instead of `detect_outliers()`'s
3-criterion rule:
- Top-20-by-raw-pvalue junction/sample pairs match R's, as a set
- Raw p-values for those shared pairs are numerically close to R's

### `test_counting.py` (4 tests, all skipped)
BAM-level counting is not implemented in protrider; kept as explicit skips rather than omitted:
- *(skipped)* Count junctions from BAM
- *(skipped)* Strand-specific counting from BAM
- *(skipped)* Min anchor length for non-split counting
- *(skipped)* PSI values from BAM counts — only the Jaccard metric is supported

### `test_unported_features.py` (4 tests, all skipped)
Features with no corresponding protrider behavior to test against; kept as explicit skips rather than omitted:
- *(skipped)* Junction distance plot — plotting is PROTRIDER/OUTRIDER-shaped
- *(skipped)* Sample results plot — plotting is PROTRIDER/OUTRIDER-shaped
- *(skipped)* Seqlevels-style conversion — no chromosome naming conversion
- *(skipped)* Full link table — no UCSC/IGV/gnomAD hyperlink column builder

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific module
pytest tests/test_pipeline_standard.py

# Run specific test
pytest tests/test_pipeline_cv.py::TestPipelineCrossValidation::test_run_with_kfold_cv

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=protrider
```

## Test Strategy

Most tests use fixed latent dimensions (e.g., `find_q_method='5'`) to:
1. Speed up test execution (no hyperparameter search)
2. Ensure reproducibility and deterministic results
3. Focus on testing pipeline logic, not dimension selection algorithms

Tests that verify automatic latent dimension selection (OHT, gs) are included to ensure these methods work correctly.

## Fixtures

Common fixtures are defined in `conftest.py`:
- `protein_intensities_path` - Sample protein data
- `covariates_path` - Sample annotations
- `protein_intensities_index_col` - Index column name
