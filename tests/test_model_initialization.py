import numpy as np
import torch

from protrider.model.model import OmicAutoencoder


def test_pca_initialization_preserves_trailing_covariate_weights():
    """PCA replaces only omics/latent weights, not conditional covariate weights."""
    model = OmicAutoencoder(in_dim=3, latent_dim=2, n_cov=2, model_type='outrider').double()
    model.encoder.model.weight.data.copy_(
        torch.tensor([[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.]])
    )
    model.decoder.model.weight.data.copy_(
        torch.tensor([[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]])
    )

    model.initialize_wPCA(
        np.array([[10., 11., 12.], [20., 21., 22.]]),
        np.zeros((1, 3)),
        n_cov=2,
    )

    torch.testing.assert_close(
        model.encoder.model.weight,
        torch.tensor([[10., 11., 12., 3., 4.], [20., 21., 22., 8., 9.]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        model.decoder.model.weight,
        torch.tensor(
            [[10., 20., 2., 3.], [11., 21., 6., 7.], [12., 22., 10., 11.]],
            dtype=torch.float64,
        ),
    )


def test_pca_initialization_for_protrider_retains_its_existing_weight_behavior():
    """The OUTRIDER PCA correction must not alter PROTRIDER initialization."""
    model = OmicAutoencoder(in_dim=3, latent_dim=2, n_cov=2, model_type='protrider').double()
    model.encoder.model.weight.data.copy_(
        torch.tensor([[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.]])
    )
    model.decoder.model.weight.data.copy_(
        torch.tensor([[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]])
    )

    model.initialize_wPCA(
        np.array([[10., 11., 12.], [20., 21., 22.]]),
        np.zeros((1, 3)),
        n_cov=2,
    )

    torch.testing.assert_close(
        model.encoder.model.weight,
        torch.tensor([[10., 11., 12., 0., 1.], [20., 21., 22., 5., 6.]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        model.decoder.model.weight,
        torch.tensor(
            [[10., 20., 0., 1.], [11., 21., 4., 5.], [12., 22., 8., 9.]],
            dtype=torch.float64,
        ),
    )
