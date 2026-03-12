"""
Tests for FRASER pipeline
"""

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from protrider.config import load_config
from protrider.pipeline import _inference
from protrider.datasets.datasets import OmicDataset
from protrider.model.model_helper import  find_latent_dim, init_model
from protrider.stats import fit_residuals

class TestPipelineFRASER:
    """Test class for fraser pipeline execution."""

    def test_run(self, config_path):
        config = load_config(config_path)
        print("config loaded.")

        fraser_dataset = OmicDataset(analysis='fraser', split_reads=[config.split_reads], unsplit_reads=[config.unsplit_reads])

        # read K and N from R
        K_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/K.tsv", sep='\t')
        N_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/N.tsv", sep='\t')

        # Check split and unsplit reads
        assert isinstance(fraser_dataset.split_reads, pd.DataFrame)
        assert isinstance(fraser_dataset.unsplit_reads, pd.DataFrame)

        # Check sample columns, k and n
        assert isinstance(fraser_dataset.K, pd.DataFrame)
        assert isinstance(fraser_dataset.N, pd.DataFrame)
        pd.testing.assert_frame_equal(fraser_dataset.K, K_from_R)
        pd.testing.assert_frame_equal(fraser_dataset.N, N_from_R)

        # Check jaccard index, filtered expression and data
        passed_expression_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/s.tsv", sep=',')[['passedExpression']]
        passed_expression_from_R['passedExpression'] = passed_expression_from_R['passedExpression'].astype(bool)
        passed_expression_python = pd.DataFrame(fraser_dataset.passed_expression).astype(bool)
        passed_expression_python.columns = ['passedExpression']
        pd.testing.assert_frame_equal(passed_expression_python, passed_expression_from_R)

        data_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/x.csv", index_col=0)
        data_from_python = pd.DataFrame(fraser_dataset.X.cpu().numpy(),index=data_from_R.index,columns=data_from_R.columns)
        # Ensure indices and columns match for comparison
        assert isinstance(fraser_dataset.X, torch.Tensor) 
        assert fraser_dataset.X.shape == (len(fraser_dataset.samples_cols), fraser_dataset.split_reads.shape[0])

        pd.testing.assert_frame_equal(data_from_python, data_from_R, check_dtype=False)
        q = find_latent_dim(fraser_dataset, method=config.find_q_method,
                        # Params for grid search method
                        inj_freq=config.inj_freq,
                        inj_mean=config.inj_mean,
                        inj_sd=config.inj_sd,
                        init_wPCA=config.init_pca,
                        n_layers=config.n_layers,
                        h_dim=config.h_dim,
                        n_epochs=config.gs_epochs if config.gs_epochs else config.n_epochs,
                        learning_rate=config.lr,
                        batch_size=config.batch_size,
                        pval_sided=config.pval_sided,
                        pval_dist=config.pval_dist,
                        out_dir=config.out_dir,
                        device=config.device_torch,
                        presence_absence=config.presence_absence,
                        lambda_bce=config.lambda_presence_absence,
                        model_type=config.analysis,
                        loss_fn=config.autoencoder_loss,
                        n_jobs=config.n_jobs,
                        )
        print(f"Selected latent dimension q: {q}")

        model = init_model(fraser_dataset, q,
                       init_wPCA=config.init_pca,
                       n_layer=config.n_layers,
                       h_dim=config.h_dim,
                       device=config.device_torch,
                       presence_absence=config.presence_absence if config.n_layers == 1 else False,
                       model_type="fraser" #TODO Config
                       )
        criterion = model.set_loss(autoencoder_loss = config.autoencoder_loss, lambda_presence_absence = config.lambda_presence_absence) 
        df_out, theta, df_presence, init_loss, init_mse_loss, init_bce_loss = _inference(fraser_dataset, model, criterion, batch_size=config.batch_size) #TODO check _inference, check in R if miu is also updated
        print("Inference completed.")

        rho_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/rho.csv", index_col=0)
        #TODO export mu
        mu, rho, _ = fit_residuals(fraser_dataset, df_out, model, config)
        print("Head of fitted mu:")
        print(mu)

        #pd.testing.assert_frame_equal(rho_df, rho_from_R, check_dtype=False)
        print("head of theta:")
        print(theta[:5])
        print("head of rho:")
        print(rho[:5])

        # Visualize
        rho_python = np.asarray(rho).flatten()
        rho_R = np.asarray(rho_from_R).flatten()
        plt.figure()
        plt.scatter(rho_R, rho_python)
        min_val = min(rho_R.min(), rho_python.min())
        max_val = max(rho_R.max(), rho_python.max())
        plt.plot([min_val, max_val], [min_val, max_val])
        plt.xlabel("rho from R")
        plt.ylabel("rho from Python")
        plt.title("NLL without L2 penalty: Python vs R")
        plt.show()


test_fraser = TestPipelineFRASER()
test_fraser.test_run("/s/project/py_fraser/PROTRIDER/config.yaml")



#residuals --> might be different for fraser, check if we can reuse the same code
#fit_residuals
# annotate genes --> check in R using a GTF file (library exists?)
#get and adjust pvals