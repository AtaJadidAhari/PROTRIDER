# from protrider.config import load_config
# from protrider.model.model_helper import find_latent_dim, init_model
# from protrider.pipeline import _inference
# from protrider.stats import fit_residuals
# import pandas as pd
# import numpy as np

# def test_best_latent_dim(fraser_dataset, config_path): #makeSimulatedFraserDataSet -> to test function correctness
#     config = load_config(config_path)
#     q = find_latent_dim(fraser_dataset, method=config.find_q_method,
#                         # Params for grid search method
#                         inj_freq=config.inj_freq,
#                         inj_mean=config.inj_mean,
#                         inj_sd=config.inj_sd,
#                         init_wPCA=config.init_pca,
#                         n_layers=config.n_layers,
#                         h_dim=config.h_dim,
#                         n_epochs=config.gs_epochs if config.gs_epochs else config.n_epochs,
#                         learning_rate=config.lr,
#                         batch_size=config.batch_size,
#                         pval_sided=config.pval_sided,
#                         pval_dist=config.pval_dist,
#                         out_dir=config.out_dir,
#                         device=config.device_torch,
#                         presence_absence=config.presence_absence,
#                         lambda_bce=config.lambda_presence_absence,
#                         model_type=config.analysis,
#                         loss_fn=config.autoencoder_loss,
#                         n_jobs=config.n_jobs,
#                         )
#     assert isinstance(q, int), "q should be an integer"
#     assert q > 1, "q should be greater than 1"
#     assert q <= len(fraser_dataset.sample_ids), "q should be less than or equal to the number of samples"
#     assert q <= fraser_dataset.X.shape[1], "q should be less than or equal to the number of junctions"

# def test_model_architecture(fraser_dataset, config_path, q):
#     pass #TODO

# def test_model_output(fraser_dataset, config_path, q):
#     config = load_config(config_path)

#     n_samples  = len(fraser_dataset.sample_ids)
#     n_junctions = fraser_dataset.split_reads.shape[0]

#     model = init_model(fraser_dataset, q,
#                        init_wPCA=config.init_pca,
#                        n_layer=config.n_layers,
#                        h_dim=config.h_dim,
#                        device=config.device_torch,
#                        presence_absence=config.presence_absence if config.n_layers == 1 else False,
#                        model_type="fraser"
#                        )
#     criterion = model.set_loss(autoencoder_loss = config.autoencoder_loss, lambda_presence_absence = config.lambda_presence_absence) 
#     df_out, theta, _, init_loss, init_mse_loss, _ = _inference(fraser_dataset, model, criterion, batch_size=config.batch_size) 


#     assert model is not None, "Model should be initialized successfully"
#     assert isinstance(df_out, pd.DataFrame), "Model output should be a DataFrame"
#     assert np.isfinite(df_out.values).all(), "Model output should not contain NaN or infinite values"
#     assert df_out.shape == fraser_dataset.centered_log_data_noNA.T.shape, "Output shape should be samples x junctions"

#     #TODO
#     assert theta is not None, "theta must not be None for fraser"
#     assert theta.shape == (n_junctions,), f"theta shape {theta.shape} != ({n_junctions},)"
#     assert ((theta > 0) & (theta < 1)).all(), "theta must be in (0, 1)"

#     assert np.isfinite(init_loss), "Initial loss should be finite"
#     assert np.isfinite(init_mse_loss), "Initial MSE loss should be finite"
#     assert init_loss > 0 , "Initial loss should be greater than 0"

# def test_fit_residuals(fraser_dataset, config_path, df_out, model):
#     config = load_config(config_path)
#     n_junctions = fraser_dataset.split_reads.shape[0]
#     n_samples   = len(fraser_dataset.sample_ids)
#     mu, rho, _, _ = fit_residuals(fraser_dataset, df_out, model, config)

#     assert mu is not None, "mu should not be None"
#     assert mu.shape == (n_junctions, n_samples), f"mu shape {mu.shape} != ({n_junctions}, {n_samples})"
#     assert np.isfinite(mu).all(), "mu should not contain NaN or infinite values"
#     assert (mu > 0).all() and (mu < 1).all(), "mu should be in (0, 1)"

#     assert rho is not None, "rho should not be None"
#     assert rho.shape == (n_junctions,), f"rho shape {rho.shape} != ({n_junctions},)"
#     assert np.isfinite(rho).all(), "rho should not contain NaN or infinite values"
#     assert (rho > 0).all() and (rho < 1).all(), "rho should be in (0, 1) for Beta-Binomial distribution"