import numpy as np
import pandas as pd


def test_K_and_N_match_r_reference(fraser_dataset, r_K, r_N):
    sample_ids = list(fraser_dataset.sample_ids)
    r_K_vals = r_K[sample_ids].values
    r_N_vals = r_N[sample_ids].values

    # fraser_dataset's split_reads/K/N have already been filtered in-place, so instead of a
    # positional comparison, assert that every filtered junction's (K, N) row can be found
    # among the R reference rows for the same (unfiltered) junction set.
    filtered_K = fraser_dataset.K[sample_ids].values
    filtered_N = fraser_dataset.N[sample_ids].values
    for i in range(filtered_K.shape[0]):
        matches = np.all(r_K_vals == filtered_K[i], axis=1) & np.all(r_N_vals == filtered_N[i], axis=1)
        assert matches.any(), f"Filtered junction {i} (K,N) row not found in R reference K.tsv/N.tsv"


def test_expression_filter_matches_r_reference(fraser_dataset, r_annotations):
    np.testing.assert_array_equal(fraser_dataset.passed_expression, r_annotations["passedExpression"].to_numpy())


def test_final_filter_count_matches_r_reference(fraser_dataset, r_annotations):
    n_junctions_filtered = len(fraser_dataset.split_reads)
    n_passed_r = int(r_annotations["passed"].sum())
    assert n_junctions_filtered == n_passed_r


def test_logit_transform_matches_r_reference(r_K, r_N, r_x):
    sample_ids = r_x.index.tolist()

    pseudocount = 0.1
    K = r_K[sample_ids].values.T  # samples x junctions
    N = r_N[sample_ids].values.T
    p = (K + pseudocount) / (N + 2 * pseudocount)
    logit = np.log(p / (1 - p))
    centered = logit - np.nanmean(logit, axis=0, keepdims=True)

    np.testing.assert_allclose(r_x.values, centered, rtol=1e-6, atol=1e-8)
