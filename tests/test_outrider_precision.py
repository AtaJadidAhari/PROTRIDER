"""Precision of preprocessing/fitting and the boundary to statistical scoring."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from pydeseq2.preprocessing import deseq2_norm
from scipy.stats import nbinom

from protrider import ProtriderConfig, run
from protrider.datasets import OutriderDataset
from protrider.datasets.outrider_oht import _oht_coefficient, _mp_cdf
from protrider.dispersions import OutriderDispersion
from protrider.model import init_model
from protrider.pipeline import _inference
from protrider.stats import get_pvals, get_pv_nb


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preprocessing_and_both_svds_follow_precision(
    gene_expression_path, gene_annotation_path, tmp_path, dtype
):
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
    sample_ids = pd.read_csv(gene_expression_path, sep='\t', nrows=0).columns.drop('geneID')
    annotations = tmp_path / 'annotations.tsv'
    pd.DataFrame({'sample_ID': sample_ids, 'AGE': np.arange(len(sample_ids)) + 20}).to_csv(
        annotations, sep='\t', index=False,
    )
    with patch('protrider.datasets.datasets.deseq2_norm', wraps=deseq2_norm) as norm:
        dataset = OutriderDataset(
            [gene_expression_path], 'geneID', gtf=gene_annotation_path,
            sa_file=str(annotations), cov_used=['AGE'], dtype=dtype, numpy_dtype=numpy_dtype,
        )
    assert norm.call_count == 2
    assert all(call.args[0].dtype == numpy_dtype for call in norm.call_args_list)
    assert dataset.raw_counts_filtered.to_numpy().dtype == np.int64
    for values in (dataset.size_factors_array, dataset.oht_size_factors,
                   dataset.fpkms.to_numpy(), dataset.normalized_log_counts.to_numpy(),
                   dataset.gene_means.to_numpy(), dataset.centered_log_data_noNA):
        assert values.dtype == numpy_dtype
    assert dataset.raw_x.dtype == dataset.X.dtype == dataset.covariates.dtype == dtype
    with patch('numpy.linalg.svd', wraps=np.linalg.svd) as svd:
        dataset.find_enc_dim_optht()
        dataset.perform_svd()
    assert svd.call_count == 2
    assert all(call.args[0].dtype == numpy_dtype for call in svd.call_args_list)
    assert dataset.oht_diagnostics['singular_values'].dtype == numpy_dtype
    assert dataset.oht_diagnostics['threshold'].dtype == numpy_dtype
    assert dataset.Vt.dtype == dataset.s.dtype == dataset.U.dtype == numpy_dtype


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize(('beta', 'expected'), [
    (0.001, 1.4166026071590645), (0.01, 1.4370270233437419),
    (0.1, 1.6087528343709245), (0.5, 2.1711323475101496), (1.0, 2.8586504329825977),
])
def test_oht_integral_agrees_with_original_double_quadrature(beta, expected, dtype):
    actual = _oht_coefficient(beta, dtype)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-3)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_oht_cdf_preserves_precision_and_endpoint_limits(dtype):
    x = np.array([0, 4], dtype=dtype)
    cumulative = _mp_cdf(x, dtype(0), dtype(4), dtype(1), dtype)
    assert cumulative.dtype == dtype
    np.testing.assert_allclose(cumulative, [0, 1], atol=1e-6)


@pytest.mark.parametrize('batch_size', [None, 3])
def test_pca_only_fits_float32_before_float64_scoring(
    gene_expression_path, gene_annotation_path, tmp_path, batch_size
):
    config = ProtriderConfig(
        out_dir=str(tmp_path), analysis='outrider', autoencoder_loss='NLL', pval_dist='nb',
        input_intensities=gene_expression_path, gtf=gene_annotation_path, index_col='geneID',
        find_q_method='5', autoencoder_training=False, batch_size=batch_size,
        outrider_precision='float32', device='cpu', calculate_one_sided_pval=True,
    )
    with patch.object(OutriderDispersion, 'fit', autospec=True, side_effect=OutriderDispersion.fit) as fit:
        result, _ = run(config)
    assert fit.call_count == 1
    assert fit.call_args.args[1].dtype == fit.call_args.args[2].dtype == torch.float32
    checkpoint = torch.load(tmp_path / 'model.pt', weights_only=False)
    assert checkpoint['theta'].dtype == checkpoint['mean_scale'].dtype == np.float32
    for frame in (result.df_expected_counts, result.dispersions, result.mu,
                  result.df_pvals, result.df_pvals_one_sided, result.df_pvals_adj,
                  result.df_Z, result.df_res, result.fc, result.log2fc):
        assert frame.to_numpy().dtype == np.float64
        assert np.isfinite(frame.to_numpy()).all()


def test_nb_scoring_uses_the_precision_of_its_inputs():
    counts = np.array([[1, 12], [7, 5], [3, 9]], dtype=np.int64)
    expected = np.array([[1.1, 11.5], [6.8, 5.3], [2.9, 9.5]], dtype=np.float32)
    theta = np.array([1.7, 13.3], dtype=np.float32)
    mean64, theta64 = expected.astype(np.float64), theta.astype(np.float64)[None, :]
    probability = theta64 / (theta64 + mean64)
    cdf = nbinom.cdf(counts, theta64, probability)
    pmf = nbinom.pmf(counts, theta64, probability)
    reference = 2 * np.minimum(np.minimum(cdf, 1 - cdf + pmf), 0.5)
    actual = get_pv_nb(counts, mean64, None, theta64.ravel())
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, reference)
    pvals, zscores = get_pvals(mean64, None, theta64.ravel(), counts, theta64.ravel(), dis='nb')
    assert pvals.dtype == zscores.dtype == np.float64
    log2fc = np.log2(counts + 1) - np.log2(mean64 + 1)
    reference_z = (log2fc - log2fc.mean(axis=0)) / log2fc.std(axis=0, ddof=1)
    np.testing.assert_allclose(zscores, reference_z, rtol=1e-13, atol=1e-13)
    assert get_pv_nb(counts, expected, None, theta).dtype == np.float32


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch_size', [None, 3])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA hardware is unavailable'
))])
def test_outrider_inference_keeps_model_on_device_and_bounds_batches(
    gene_expression_path, gene_annotation_path, dtype, batch_size, device
):
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
    dataset = OutriderDataset(
        [gene_expression_path], 'geneID', gtf=gene_annotation_path,
        dtype=dtype, numpy_dtype=numpy_dtype, device=torch.device(device),
    )
    model = init_model(dataset, 5, model_type='outrider', device=torch.device(device))
    criterion = model.set_loss('NLL')
    with (
        patch.object(model, 'to', side_effect=AssertionError('model device transfer')),
        patch.object(criterion, 'forward', wraps=criterion.forward) as loss,
    ):
        df_out, theta, _, nll, _, _ = _inference(dataset, model, criterion, batch_size=batch_size)
    assert next(model.parameters()).device.type == device
    assert model.latent_values.shape[0] == len(dataset.X)
    assert df_out.to_numpy().dtype == theta.dtype == numpy_dtype
    assert np.isfinite(nll)
    assert all(call.args[1].shape[0] <= (batch_size or len(dataset.X)) for call in loss.call_args_list)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('fit_mean_scale', [False, True])
def test_batched_theta_optimizer_uses_full_cohort_objective_and_gradients(dtype, fit_mean_scale):
    counts = torch.tensor([[0, 10], [5, 2], [25, 0], [2, 30], [3, 8]], dtype=dtype)
    expected = counts * 0.7 + 2
    captures = []

    def inspect_step(optimizer, closure):
        parameters = optimizer.param_groups[0]['params']
        assert all(parameter.dtype == dtype for parameter in parameters)
        objective = closure()
        assert objective.dtype == dtype
        captures.append((objective.detach(), [p.grad.clone() for p in parameters]))

    with patch.object(torch.optim.LBFGS, 'step', inspect_step):
        for batch_size in (None, 2):
            OutriderDispersion().fit(
                counts, expected, fit_mean_scale=fit_mean_scale, batch_size=batch_size,
            )
    full_loss, full_grads = captures[0]
    chunk_loss, chunk_grads = captures[1]
    torch.testing.assert_close(chunk_loss, full_loss)
    for actual, reference in zip(chunk_grads, full_grads):
        torch.testing.assert_close(actual, reference)
