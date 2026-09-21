"""Numerical and sampling invariants for OUTRIDER performance optimizations."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from protrider.datasets.datasets import OutriderDataset
from protrider.dispersions import NegativeBinomialDistribution, OutriderDispersion
from protrider.estimate_theta_robust_moments import estimate_theta_robust_moments


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware is unavailable"
))]


class UncachedNB(NegativeBinomialDistribution):
    """Original likelihood evaluation, including all count-only terms."""

    def loss(self, counts, theta, mu):
        term_lgamma = torch.lgamma(counts + theta) - torch.lgamma(theta) - torch.lgamma(counts + 1)
        log_prob = (term_lgamma + torch.xlogy(theta, theta)
                    + torch.xlogy(counts, mu) - torch.xlogy(counts + theta, theta + mu))
        return -torch.sum(torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=-1e20))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fit_mean_scale", [False, True])
def test_cached_likelihood_preserves_objective_gradients_and_fit(device, dtype, fit_mean_scale):
    counts = torch.tensor([[0, 10, 700], [5, 2, 900], [25, 0, 300], [2, 30, 1100]],
                          device=device, dtype=dtype)
    expected = counts * 0.7 + 2
    theta = torch.tensor([[2, 5, 15]], device=device, dtype=dtype, requires_grad=True)
    scale = torch.ones(1, 3, device=device, dtype=dtype, requires_grad=fit_mean_scale)
    mean = expected * scale
    parameters = (theta, scale) if fit_mean_scale else (theta,)
    reference = UncachedNB().loss(counts, theta, mean)
    cached = NegativeBinomialDistribution().loss(
        counts, theta, mean, count_lgamma=torch.lgamma(counts + 1),
        count_log_mu=None if fit_mean_scale else torch.xlogy(counts, expected),
    )
    torch.testing.assert_close(cached, reference, rtol=0, atol=0)
    reference_grads = torch.autograd.grad(reference, parameters, retain_graph=True)
    cached_grads = torch.autograd.grad(cached, parameters)
    for actual, wanted in zip(cached_grads, reference_grads):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    optimized = OutriderDispersion()
    original = OutriderDispersion(UncachedNB())
    # Reuse counts across fits, but change predictions: only count constants
    # and the original moment estimate may survive between fits.
    for prediction in (expected, expected * 1.2):
        optimized.fit(counts, prediction, max_iter=20, fit_mean_scale=fit_mean_scale)
        original.fit(counts, prediction, max_iter=20, fit_mean_scale=fit_mean_scale)
        torch.testing.assert_close(optimized.theta, original.theta, rtol=0, atol=0)
        if fit_mean_scale:
            torch.testing.assert_close(optimized.mean_scale, original.mean_scale, rtol=0, atol=0)


def test_count_cache_reuses_estimate_and_invalidates_changed_counts_or_bounds():
    counts = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    dispersion = OutriderDispersion()
    with patch("protrider.dispersions.estimate_theta_robust_moments",
               wraps=estimate_theta_robust_moments) as estimate:
        dispersion.fit(counts, counts + 1, max_iter=2)
        dispersion.fit(counts, counts + 2, max_iter=2)
        assert estimate.call_count == 1
        counts[0, 0] = 40
        dispersion.fit(counts, counts + 2, max_iter=2)
        assert estimate.call_count == 2
        dispersion.fit(counts, counts + 2, max_iter=2, upper_bound=500)
        assert estimate.call_count == 3
        dispersion.fit(counts.clone(), counts + 2, max_iter=2, upper_bound=500)
        assert estimate.call_count == 4
    fresh = OutriderDispersion()
    fresh.fit(counts, counts + 2, max_iter=2, upper_bound=500)
    torch.testing.assert_close(dispersion.theta, fresh.theta, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [3, 7])
def test_tensor_batches_preserve_shuffle_order_and_rng(device, batch_size):
    dataset = OutriderDataset.__new__(OutriderDataset)
    dataset.X = torch.arange(28, device=device, dtype=torch.float32).reshape(7, 4)
    dataset.raw_x = dataset.X + 1
    dataset.torch_mask = torch.zeros_like(dataset.X, dtype=torch.bool)
    dataset.covariates = dataset.X[:, :2]
    dataset.omic_means_torch = dataset.X.mean(dim=0)
    dataset.size_factors = torch.ones(7, 1, device=device)
    original = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batched = DataLoader(dataset, batch_size=None,
                         sampler=BatchSampler(RandomSampler(dataset), batch_size, False))

    torch.manual_seed(123)
    expected = [list(original) for _ in range(3)]
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)
    actual = [list(batched) for _ in range(3)]
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for expected_epoch, actual_epoch in zip(expected, actual):
        assert len(expected_epoch) == len(actual_epoch)
        for expected_batch, actual_batch in zip(expected_epoch, actual_epoch):
            # Gene means are shared metadata and no longer stacked per sample.
            for field in (0, 1, 2, 4, 5):
                torch.testing.assert_close(actual_batch[field], expected_batch[field], rtol=0, atol=0)


def test_svd_reuse_detects_in_place_changes():
    dataset = OutriderDataset.__new__(OutriderDataset)
    dataset.centered_log_data_noNA = np.arange(28, dtype=np.float32).reshape(7, 4)
    with patch("numpy.linalg.svd", wraps=np.linalg.svd) as svd:
        dataset.perform_svd()
        first = dataset.Vt.copy()
        dataset.perform_svd()
        assert svd.call_count == 1
        np.testing.assert_array_equal(dataset.Vt, first)
        dataset.centered_log_data_noNA[0, 0] = 100
        dataset.perform_svd()
        assert svd.call_count == 2
    expected = np.linalg.svd(dataset.centered_log_data_noNA, full_matrices=False)
    np.testing.assert_array_equal(dataset.Vt, expected[2])
