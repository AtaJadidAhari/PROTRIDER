import numpy as np
import torch

from protrider.model.model_helper import init_model
from protrider.pipeline import _inference

# FRASER-R's config for this dataset used implementation='PCA' with useOHTtoObtainQ=true,
# which resolves to q=2 for these 10 samples (dataset.find_enc_dim_optht()).
Q = 2


def test_init_model_pca_reconstruction_matches_r_predicted_means(fraser_dataset_unfiltered, r_predicted_means):
    """FRASER-R's fitPCA (implementation='PCA', subset=FALSE) fits a single-pass PCA:
    E = D = top-q loadings of the centered logit matrix, b = per-junction mean of the raw
    logit matrix, predicted mu = sigmoid(H @ D.T + b). Our init_model(init_wPCA=True,
    n_layers=1) with a fresh (untrained) inference pass performs exactly this reconstruction,
    so its output should match FRASER-R's exported predicted_means.csv almost exactly."""
    ds = fraser_dataset_unfiltered
    model = init_model(ds, Q, init_wPCA=True, n_layer=1, h_dim=None,
                        device=torch.device("cpu"), presence_absence=False, model_type="fraser")
    criterion = model.set_loss(autoencoder_loss="NLL", lambda_presence_absence=0.5)

    df_out, theta, df_presence, loss, mse, bce = _inference(ds, model, criterion, batch_size=None)
    mu_pred = torch.sigmoid(torch.tensor(df_out.values)).numpy()  # samples x junctions

    pm_ref = r_predicted_means[ds.sample_ids.tolist()].values.T  # -> samples x junctions, same row order as K.tsv

    np.testing.assert_allclose(mu_pred, pm_ref, atol=1e-8)


def test_dispersion_fit_correlates_with_r_rho(fraser_dataset_unfiltered, r_rho):
    """R's updateRho() and our FraserDispersion.fit() both fit a Beta-Binomial dispersion per
    junction via numerical optimization, but with different objectives/regularization, so an
    exact match isn't expected. It should still be strongly correlated with FRASER-R's rho.csv,
    fit on the same (mu, K, N)."""
    ds = fraser_dataset_unfiltered
    model = init_model(ds, Q, init_wPCA=True, n_layer=1, h_dim=None,
                        device=torch.device("cpu"), presence_absence=False, model_type="fraser")
    criterion = model.set_loss(autoencoder_loss="NLL", lambda_presence_absence=0.5)
    df_out, theta, df_presence, loss, mse, bce = _inference(ds, model, criterion, batch_size=None)

    model.fit_dispersion(
        torch.tensor(ds.K.values, dtype=torch.float64),
        torch.tensor(ds.N.values, dtype=torch.float64),
        torch.tensor(df_out.values, dtype=torch.float64),
        max_iter=100,
    )
    _, rho_fit = model.get_dispersion_parameters()

    corr = np.corrcoef(rho_fit, r_rho.values.squeeze())[0, 1]
    assert corr >= 0.6, f"Fitted rho should correlate with FRASER-R's rho.csv, got corr={corr:.3f}"
