import numpy as np
import pytest
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


def test_gene_level_pvals_shape_matches_number_of_genes(r_annotations, r_pvals):
    """Gene-level p-value table has one row per gene, junction-level table has one row per junction."""
    group_ids = r_annotations["hgnc_symbol"]

    pvals_adj, gene_level_info = adjust_pvals(r_pvals, method="holm", group_ids=group_ids, aggregate=True, n_jobs=1)

    # A junction spanning multiple overlapping genes (semicolon-joined hgnc_symbol) is
    # exploded into one row per individual gene, so the gene count is the number of
    # distinct individual gene symbols, not the number of distinct (possibly combined)
    # group-id strings.
    n_genes = group_ids.dropna().str.split(";").explode().nunique()
    assert pvals_adj.shape[0] == n_genes
    assert gene_level_info["junction_pvals_adj"].shape[0] == r_pvals.shape[0]


# ── Not yet implemented ──────────────────────────────────────────────────

def test_betabinom_pvalue_randomization():
    """Placeholder for testing randomization used in Beta-Binomial p-value calculation."""
    pytest.skip("nothing to test yet; placeholder")


def test_padj_values_with_rho_cutoff_propagates_na():
    """Junctions/genes whose fitted dispersion exceeds a cutoff should be NA'd out before
    FDR correction. protrider.stats.adjust_pvals has no rhoCutoff parameter yet."""
    pytest.skip("adjust_pvals has no rhoCutoff parameter yet; dispersion-based NA masking not implemented")


def test_padj_values_on_subset_of_genes():
    """FDR correction restricted to a per-sample list of candidate genes (e.g. from a gene panel)
    is not implemented in protrider.stats yet."""
    pytest.skip("no per-sample gene-subset FDR correction implemented yet")
