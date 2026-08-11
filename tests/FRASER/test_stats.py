import numpy as np
from protrider.stats import get_pvals, adjust_pvals


def test_get_pvals_bb_matches_r_reference(r_K, r_N, r_predicted_means, r_rho, r_pvals):
    sample_ids = r_predicted_means.columns.tolist()
    x_true = r_K[sample_ids].values.T  # samples x junctions
    res = r_N[sample_ids].values.T

    pvals, z = get_pvals(x_true=x_true, res=res, mu=r_predicted_means.values, sigma=r_rho.values.squeeze(),
                          df0=None, how="two-sided", theta=None, dis="bb", n_jobs=1)

    assert z is None, "Beta-Binomial p-values do not produce z-scores"
    np.testing.assert_allclose(pvals.T, r_pvals.values, atol=1e-6, equal_nan=True)


def test_adjust_pvals_holm_matches_r_reference(r_annotations, r_pvals, r_pvals_adj):
    _ , gene_level_info = adjust_pvals(r_pvals, method="holm", group_ids=r_annotations["hgnc_symbol"], aggregate=True, n_jobs=1)

    junction_pvals_adj = gene_level_info["junction_pvals_adj"]
    np.testing.assert_allclose(junction_pvals_adj.values, r_pvals_adj.values, atol=1e-4, equal_nan=True)
