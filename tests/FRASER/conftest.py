import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from protrider.datasets.datasets import OmicDataset

DATA_DIR = Path(__file__).parent.parent.parent / "sample_data/fraser"


@pytest.fixture(scope="session")
def fraser_dataset(split_reads_path, unsplit_reads_path):
    return OmicDataset(analysis='fraser',split_reads=split_reads_path,unsplit_reads=unsplit_reads_path)


# ── R reference fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def r_x():
    """R reference: logit-transformed centered input (samples × junctions)."""
    return pd.read_csv(DATA_DIR / "x.csv", index_col=0)


@pytest.fixture(scope="session")
def r_predicted_means():
    """R reference: autoencoder predicted means (junctions × samples)."""
    return pd.read_csv(DATA_DIR / "predicted_means.csv", index_col=0)


@pytest.fixture(scope="session")
def r_rho():
    """R reference: BetaBinomial dispersion per junction, shape (junctions, 1)."""
    return pd.read_csv(DATA_DIR / "rho.csv", index_col=0)


@pytest.fixture(scope="session")
def r_pvals():
    """R reference: raw p-values (junctions × samples)."""
    return pd.read_csv(DATA_DIR / "pVals.csv", index_col=0)


@pytest.fixture(scope="session")
def r_pvals_adj():
    """R reference: adjusted p-values (junctions × samples)."""
    return pd.read_csv(DATA_DIR / "pValsAdj.csv", index_col=0)


@pytest.fixture(scope="session")
def r_annotations():
    """R reference: junction annotations including hgnc_symbol and annotatedJunction."""
    return pd.read_csv(DATA_DIR / "junction_annotations.csv", index_col=0)


# ── Upstream inputs for each test step (fed from R, not Python) ───────────────
# Tests receive R reference data as their inputs so a failure in one step
# does not cascade into the next test.

@pytest.fixture(scope="session")
def q():
    """Latent dim: derived from R reference X (n_samples - 1)."""
    return 2

@pytest.fixture(scope="session")
def df_out(r_predicted_means):
    """R predicted means passed as df_out to fit_residuals."""
    return r_predicted_means.copy()


@pytest.fixture(scope="session")
def mu(r_predicted_means):
    """R predicted means as numpy array (junctions × samples)."""
    return r_predicted_means.values


@pytest.fixture(scope="session")
def rho(r_rho):
    """R dispersion as 1-D numpy array (junctions,)."""
    return r_rho.values.squeeze()


@pytest.fixture(scope="session")
def pvals(r_pvals):
    """R raw p-values DataFrame passed to adjusted-pvals and gene-level tests."""
    return r_pvals.copy()
