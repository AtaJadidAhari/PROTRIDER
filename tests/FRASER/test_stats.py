from protrider.config import load_config
from protrider.stats import get_pvals, get_pvals_by_gene, adjust_pvals, get_delta_psi
import pandas as pd
import numpy as np

def test_pvals(fraser_dataset, config_path, rho, mu):
    config = load_config(config_path)
    pvals, z = get_pvals(x_true=fraser_dataset.K.values.T,
                         res=fraser_dataset.N.values.T,
                         mu=mu,
                         sigma=rho,
                         df0=None,
                         how='two-sided',
                         theta=None,
                         dis='bb',
                         n_jobs=config.n_jobs)
    pvals_df = pd.DataFrame(pvals).T
    assert pvals_df.shape == fraser_dataset.centered_log_data_noNA.T.shape, f"pvals shape {pvals_df.shape} != {fraser_dataset.centered_log_data_noNA.T.shape}"
    finite = pvals_df.values[np.isfinite(pvals_df.values)]
    assert (finite >= 0).all() and (finite <= 1).all(), "p-values must be in [0, 1]"
    assert np.isfinite(pvals_df.values).any(), "p-values should not be all NaN or infinite"
    assert pvals.shape == z.shape, f"pvals shape {pvals.shape} != z shape {z.shape}"

def test_pvals_by_gene(fraser_dataset, pvals):
    pvals_by_gene = get_pvals_by_gene(pvals, fraser_dataset.intron_ranges["hgnc_symbol"])
    assert isinstance(pvals_by_gene, pd.DataFrame), "pvals_by_gene should be a DataFrame"
    assert pvals_by_gene.shape == pvals.shape, f"pvals_by_gene shape {pvals_by_gene.shape} != pvals shape {pvals.shape}"
    finite = np.isfinite(pvals.values.astype(float)) & np.isfinite(pvals_by_gene.values.astype(float))
    assert (pvals_by_gene.values[finite] <= pvals.values[finite] + 1e-9).all(), "gene-min p-values must be ≤ raw p-values"


def test_adjust_pvals(fraser_dataset, pvals):
    adjusted_pvals = adjust_pvals(pvals, method='holm', group_ids=fraser_dataset.intron_ranges["hgnc_symbol"])
    assert isinstance(adjusted_pvals, pd.DataFrame), "adjusted_pvals should be a DataFrame"
    assert adjusted_pvals.shape == pvals.shape, f"adjusted_pvals shape {adjusted_pvals.shape} != pvals shape {pvals.shape}"
    finite = np.isfinite(adjusted_pvals.values) & np.isfinite(pvals.values)
    assert (adjusted_pvals.values[finite] >= pvals.values[finite] - 1e-9).all(), "adjusted p-values must be ≥ raw p-values"
    finite = adjusted_pvals.values[np.isfinite(adjusted_pvals.values)]
    assert (finite >= 0).all() and (finite <= 1).all(), "adjusted p-values must be in [0, 1]"

def test_delta_psi(fraser_dataset, mu):
    delta = get_delta_psi(fraser_dataset.K, fraser_dataset.N, mu)
    finite = np.asarray(delta, dtype=float)[np.isfinite(np.asarray(delta, dtype=float))]
    assert (finite > -1).all() and (finite < 1).all(), "delta_psi values must be in (-1, 1)"