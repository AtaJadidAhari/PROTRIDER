import pytest
import pandas as pd
from pathlib import Path

from protrider.datasets.datasets import FraserDataset

ROOT = Path(__file__).parent.parent.parent  # PROTRIDER/
DATA_DIR = ROOT / "sample_data/fraser"
GTF_PATH = ROOT / "sample_data/gencode_annotation_trunc.gtf"

# Fixed filter params used consistently across dataset construction and reference comparisons.
MIN_EXPRESSION_IN_ONE_SAMPLE = 20
QUANTILE = 0.95
QUANTILE_MIN_EXPRESSION = 10
PSEUDOCOUNT = 0.1
MIN_DELTA_PSI = 0.05


@pytest.fixture(scope="session")
def split_reads_path():
    return str(DATA_DIR / "split_reads.tsv")


@pytest.fixture(scope="session")
def unsplit_reads_path():
    return str(DATA_DIR / "unsplit_reads.tsv")


@pytest.fixture(scope="session")
def fraser_dataset(split_reads_path, unsplit_reads_path):
    """FraserDataset built from the real sample_data/fraser sample, without gene annotation."""
    return FraserDataset(
        split_reads=[split_reads_path],
        unsplit_reads=[unsplit_reads_path],
        minExpressionInOneSample=MIN_EXPRESSION_IN_ONE_SAMPLE,
        quantile=QUANTILE,
        quantileMinExpression=QUANTILE_MIN_EXPRESSION,
        pseudocount=PSEUDOCOUNT,
        min_delta_psi=MIN_DELTA_PSI,
    )


@pytest.fixture(scope="session")
def fraser_dataset_unfiltered(split_reads_path, unsplit_reads_path):
    """FraserDataset with all filters disabled, keeping all 982 junctions.

    PCA fitting fits the encoder/decoder on the *full* junction set, not just the ones
    passing the expression/variability filters (those filters only decide which junctions
    get tested downstream). Reproducing the reference predicted means/dispersion therefore
    requires fitting on this unfiltered dataset, in the same row order as the reference files.
    """
    return FraserDataset(
        split_reads=[split_reads_path],
        unsplit_reads=[unsplit_reads_path],
        minExpressionInOneSample=0,
        quantile=0.0,
        quantileMinExpression=0,
        pseudocount=PSEUDOCOUNT,
        min_delta_psi=0.0,
    )


@pytest.fixture(scope="session")
def fraser_dataset_annotated(split_reads_path, unsplit_reads_path):
    """Same dataset, but with gene annotation applied (needed for gene-level tests)."""
    return FraserDataset(
        split_reads=[split_reads_path],
        unsplit_reads=[unsplit_reads_path],
        minExpressionInOneSample=MIN_EXPRESSION_IN_ONE_SAMPLE,
        quantile=QUANTILE,
        quantileMinExpression=QUANTILE_MIN_EXPRESSION,
        pseudocount=PSEUDOCOUNT,
        min_delta_psi=MIN_DELTA_PSI,
        gtf=str(GTF_PATH),
    )


# ── Reference fixtures (full, unfiltered set of 982 junctions) ──────────────

@pytest.fixture(scope="session")
def r_K():
    """Reference split-read counts K, all junctions (junctions x samples)."""
    return pd.read_csv(DATA_DIR / "K.tsv", sep="\t")


@pytest.fixture(scope="session")
def r_N():
    """Reference Jaccard denominator N, all junctions (junctions x samples)."""
    return pd.read_csv(DATA_DIR / "N.tsv", sep="\t")


@pytest.fixture(scope="session")
def r_annotations():
    """Reference per-junction filter flags and gene annotations, all junctions."""
    return pd.read_csv(DATA_DIR / "junction_annotations.csv", index_col=0)


@pytest.fixture(scope="session")
def r_x():
    """Reference centered logit-transformed input matrix, all junctions (samples x junctions)."""
    return pd.read_csv(DATA_DIR / "x.csv", index_col=0)


@pytest.fixture(scope="session")
def r_predicted_means():
    """Reference PCA-fitted predicted means, all junctions (junctions x samples)."""
    return pd.read_csv(DATA_DIR / "predicted_means.csv", index_col=0)


@pytest.fixture(scope="session")
def r_rho():
    """Reference fitted Beta-Binomial dispersion, all junctions (junctions x 1)."""
    return pd.read_csv(DATA_DIR / "rho.csv", index_col=0)


@pytest.fixture(scope="session")
def r_pvals():
    """Reference raw Beta-Binomial p-values, all junctions (junctions x samples)."""
    return pd.read_csv(DATA_DIR / "pVals.csv")


@pytest.fixture(scope="session")
def r_pvals_adj():
    """Reference Holm/BY-adjusted p-values, all junctions (junctions x samples)."""
    return pd.read_csv(DATA_DIR / "pValsAdj.csv")
