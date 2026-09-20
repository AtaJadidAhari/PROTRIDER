import numpy as np

from protrider.datasets import OutriderDataset
from protrider.datasets.covariates import parse_covariates
from protrider.datasets.read_inputs import read
import pandas as pd


def test_read(protein_intensities_path, protein_intensities_index_col):
    protein_intensities_path = [protein_intensities_path]
    df = read(
        protein_intensities_path, protein_intensities_index_col)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'sampleID'
    assert df.shape == (64, 200)


def test_parse_categorical_covariates(categorical_covariates, covariates_path, protein_intensities_path, protein_intensities_index_col):
    """Test basic integration between protein intensities and categorical covariates."""
    protein_intensities_path = [protein_intensities_path]
    protein_intensities = read(
        protein_intensities_path, protein_intensities_index_col)
    covariates, centered_covariates_noNA = parse_covariates(covariates_path, categorical_covariates, protein_intensities.index)
    assert covariates.shape[0] == protein_intensities.shape[0]
    assert centered_covariates_noNA.shape[0] == protein_intensities.shape[0]


def test_outrider_dataset_uses_centered_covariates(gene_expression_path, gene_annotation_path, tmp_path):
    """OUTRIDER conditions on centered numerical covariates."""
    sample_ids = pd.read_csv(gene_expression_path, sep='\t', nrows=0).columns.drop('geneID')
    annotation_path = tmp_path / 'sample_annotations.tsv'
    pd.DataFrame({'sample_ID': sample_ids, 'AGE': np.arange(len(sample_ids)) + 20}).to_csv(
        annotation_path, sep='\t', index=False,
    )

    dataset = OutriderDataset(
        [gene_expression_path],
        'geneID',
        sa_file=str(annotation_path),
        cov_used=['AGE'],
        gtf=gene_annotation_path,
    )
    raw_covariates, centered_covariates = parse_covariates(
        annotation_path, ['AGE'], dataset.data.index,
    )

    np.testing.assert_allclose(dataset.covariates.cpu().numpy(), centered_covariates)
    np.testing.assert_allclose(dataset.raw_covariates.cpu().numpy(), raw_covariates)
