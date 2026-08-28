"""Compares the Python FRASER pipeline against a real R fraser_run_timed.R run.

"""
import gzip

import numpy as np
import pandas as pd
import pytest

from protrider import ProtriderConfig, run

from conftest import (
    DATA_DIR,
    GTF_PATH,
    MIN_EXPRESSION_IN_ONE_SAMPLE,
    QUANTILE,
    QUANTILE_MIN_EXPRESSION,
    PSEUDOCOUNT,
    MIN_DELTA_PSI,
)

TOP_N = 20
JOIN_COLS = ["seqnames", "start", "end", "strand", "sampleID"]


@pytest.fixture(scope="module")
def r_top_outliers_by_pvalue():
    """Top TOP_N (junction, sample) pairs from the R run, by raw p-value (column: pValue)."""
    path = DATA_DIR / "r_results_all_junctions_fraser_run_timed.tsv.gz"
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t")
    return df.sort_values("pValue").head(TOP_N).reset_index(drop=True)


@pytest.fixture(scope="module")
def python_result(split_reads_path, unsplit_reads_path, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("fraser_r_vs_py")
    config = ProtriderConfig(
        out_dir=str(tmp_dir),
        analysis="fraser",
        input_intensities="unused",
        split_reads=split_reads_path,
        unsplit_reads=unsplit_reads_path,
        gtf=str(GTF_PATH),
        min_expression_in_one_sample=MIN_EXPRESSION_IN_ONE_SAMPLE,
        quantile_for_filtering=QUANTILE,
        quantile_min_expression=QUANTILE_MIN_EXPRESSION,
        pseudocount=PSEUDOCOUNT,
        min_delta_psi=MIN_DELTA_PSI,
        find_q_method="1",  # matches R's actual OHT-selected q here 
        init_pca=True,
        autoencoder_training=False,  
        autoencoder_loss="BBL",
        pval_dist="bb",
        pval_adj="holm",
        device="cpu",
        n_jobs=1,
        verbose=False,
    )
    result, _ = run(config)
    return result


@pytest.fixture(scope="module")
def python_top_outliers_by_pvalue(python_result, tmp_path_factory):
    """Top TOP_N (junction, sample) pairs from the Python pipeline, by raw p-value. Uses the
    unfiltered long table from save(..., include_all=True), ignoring detect_outliers()'s
    3-criterion flag entirely."""
    tmp_dir = tmp_path_factory.mktemp("fraser_r_vs_py_save")
    python_result.save(str(tmp_dir), format="long", analysis="fraser", include_all=True)
    df_all = pd.read_csv(tmp_dir / "fraser_summary_all_junctions.csv.gz")
    return df_all.sort_values("JUNCTION_PVALUE").head(TOP_N).reset_index(drop=True)


def test_python_top_outliers_match_r_by_junction_and_sample(python_top_outliers_by_pvalue, r_top_outliers_by_pvalue):
    """Top-20-by-raw-pvalue (junction, sample) pairs match as a set; order isn't required to
    match since near-tied pairs can trade ranks between implementations."""
    python_pairs = set(map(tuple, python_top_outliers_by_pvalue[JOIN_COLS].to_numpy()))
    r_pairs = set(map(tuple, r_top_outliers_by_pvalue[JOIN_COLS].to_numpy()))

    assert python_pairs == r_pairs, (
        f"Top {TOP_N} outliers by raw p-value differ between Python and R.\n"
        f"Only in Python: {python_pairs - r_pairs}\n"
        f"Only in R: {r_pairs - python_pairs}"
    )


def test_python_top_outlier_pvalues_close_to_r(python_top_outliers_by_pvalue, r_top_outliers_by_pvalue):
    """Raw p-values for the shared top-20 pairs are close, not bit-identical (Beta-Binomial
    exact test uses randomized tie-breaking)."""
    merged = python_top_outliers_by_pvalue.merge(
        r_top_outliers_by_pvalue, on=JOIN_COLS, suffixes=("_py", "_r"),
    )
    assert len(merged) == TOP_N, "expected every python top-20 row to have a matching R row"

    np.testing.assert_allclose(
        merged["JUNCTION_PVALUE"].to_numpy(), merged["pValue"].to_numpy(), atol=0.005,
    )
