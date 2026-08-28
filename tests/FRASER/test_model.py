import numpy as np
import torch
from optht.optht import _optimal_SVHT_coef_sigma_known, _median_marcenko_pastur
from protrider.model.model_helper import init_model
from protrider.pipeline import _inference

Q = 2


def test_init_model_pca_reconstruction_matches_r_predicted_means(fraser_dataset_unfiltered, r_predicted_means):
    ds = fraser_dataset_unfiltered
    model = init_model(ds, Q, init_wPCA=True, n_layer=1, h_dim=None,
                        device=torch.device("cpu"), presence_absence=False, model_type="fraser")
    criterion = model.set_loss(autoencoder_loss="BBL", lambda_presence_absence=0.5)

    df_out, theta, df_presence, loss, mse, bce = _inference(ds, model, criterion, batch_size=None)
    mu_pred = torch.sigmoid(torch.tensor(df_out.values)).numpy()  # samples x junctions

    pm_ref = r_predicted_means[ds.sample_ids.tolist()].values.T  # -> samples x junctions, same row order as K.tsv

    np.testing.assert_allclose(mu_pred, pm_ref, atol=1e-6)


def test_dispersion_fit_correlates_with_r_rho(fraser_dataset_unfiltered, r_rho):
    ds = fraser_dataset_unfiltered
    model = init_model(ds, Q, init_wPCA=True, n_layer=1, h_dim=None,
                        device=torch.device("cpu"), presence_absence=False, model_type="fraser")
    criterion = model.set_loss(autoencoder_loss="BBL", lambda_presence_absence=0.5)
    df_out, theta, df_presence, loss, mse, bce = _inference(ds, model, criterion, batch_size=None)

    model.fit_dispersion(
        torch.tensor(ds.K.values, dtype=torch.float32),
        torch.tensor(ds.N.values, dtype=torch.float32),
        torch.tensor(df_out.values, dtype=torch.float32),
        max_iter=100,
    )
    _, rho_fit = model.get_dispersion_parameters()

    corr = np.corrcoef(rho_fit, r_rho.values.squeeze())[0, 1]
    assert corr >= 0.999, f"Fitted rho should correlate with FRASER-R's rho.csv, got corr={corr:.3f}"


def test_optimal_svht_coef_matches_r_reference():
    """Gavish-Donoho coefficient w(beta) for known noise level, at aspect ratios beta=0.5 and beta=0.1."""
    np.testing.assert_allclose(_optimal_SVHT_coef_sigma_known(0.5), 1.98, atol=0.01)
    np.testing.assert_allclose(_optimal_SVHT_coef_sigma_known(0.1), 1.58, atol=0.01)


def test_median_marchenko_pastur_matches_r_reference():
    """Reproduces Table IV of Gavish & Donoho (2014) by combining optimalSVHTCoef with the
    median Marchenko-Pastur quantile for two (n, m) shapes with beta = min(n,m)/max(n,m)."""
    beta_100_200 = 100 / 200
    beta_100_1000 = 100 / 1000

    coef_1 = _optimal_SVHT_coef_sigma_known(beta_100_200) / np.sqrt(_median_marcenko_pastur(beta_100_200))
    coef_2 = _optimal_SVHT_coef_sigma_known(beta_100_1000) / np.sqrt(_median_marcenko_pastur(beta_100_1000))

    np.testing.assert_allclose(coef_1, 2.1711, atol=0.0001)
    np.testing.assert_allclose(coef_2, 1.6089, atol=0.001)


def test_find_enc_dim_optht_recovers_known_rank(fraser_dataset_unfiltered):
    """On a regular (non-trivial) dataset, optimal hard thresholding should pick a latent
    dimension >= 2 (FraserDataset.find_enc_dim_optht() clamps to 2 for tiny/degenerate data)."""
    ds = fraser_dataset_unfiltered
    q = ds.find_enc_dim_optht()
    assert q >= 2
    assert isinstance(q, (int, np.integer))
