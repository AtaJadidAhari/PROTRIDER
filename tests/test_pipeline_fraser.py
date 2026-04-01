"""
Tests for FRASER pipeline
"""

import pandas as pd
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
from protrider.config import load_config
from protrider.pipeline import _inference
from protrider.datasets.datasets import OmicDataset
from protrider.model.model_helper import  find_latent_dim, init_model
from protrider.stats import fit_residuals, get_pvals
from scipy.special import expit


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
        data_from_python = pd.DataFrame(fraser_dataset.centered_log_data_noNA, index=data_from_R.index, columns=data_from_R.columns)
        # Ensure indices and columns match for comparison
        assert isinstance(fraser_dataset.centered_log_data_noNA, np.ndarray)
        assert fraser_dataset.centered_log_data_noNA.shape == (len(fraser_dataset.samples_cols), fraser_dataset.split_reads.shape[0])

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
                       model_type="fraser"
                       )
        criterion = model.set_loss(autoencoder_loss = config.autoencoder_loss, lambda_presence_absence = config.lambda_presence_absence) 
        df_out, theta, df_presence, init_loss, init_mse_loss, init_bce_loss = _inference(fraser_dataset, model, criterion, batch_size=config.batch_size) #TODO check _inference, check in R if miu is also updated
        print("Inference completed.")

        rho_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/rho.csv", index_col=0)

        mu, rho, _, _ = fit_residuals(fraser_dataset, df_out, model, config)
        #print("Head of fitted mu:")
        #print(mu)

        #pd.testing.assert_frame_equal(rho_df, rho_from_R, check_dtype=False)
        #print("head of theta:")
        #print(theta[:5])
        #print("head of rho:")
        #print(rho[:5])

        # Visualize
        """rho_python = np.asarray(rho).flatten()
        rho_R = np.asarray(rho_from_R).flatten()
        plt.figure()
        plt.scatter(rho_R, rho_python)
        min_val = min(rho_R.min(), rho_python.min())
        max_val = max(rho_R.max(), rho_python.max())
        plt.plot([min_val, max_val], [min_val, max_val])
        plt.xlabel("rho from R")
        plt.ylabel("rho from Python")
        plt.title("NLL without L2 penalty: Python vs R")
        plt.show()"""
        fraser_dataset.annotate_junctions("/s/project/py_fraser/PROTRIDER/sample_data/gencode_annotation_trunc.gtf")
        print("Annotation completed.")
        annotated_from_python = fraser_dataset.intron_ranges.copy()

        annotated_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/junction_annotations.csv", index_col=0)
        annotated_from_R = annotated_from_R[["startID", "endID", "hgnc_symbol", "annotatedJunction"]].reset_index(drop=True)
        annotated_from_R.columns = ["StartId", "EndId", "hgnc_symbol", "annotatedJunction"]  
        
        def sort_genes(val):
            if isinstance(val, str):
                return ";".join(sorted(val.split(";")))
            return val
        def normalize_genes(val):
            if isinstance(val, str):
                genes = [re.sub(r'_\d+$', '', g) for g in val.split(";")]
                return ";".join(sorted(set(genes)))
            return val
        
        annotated_from_python["hgnc_symbol"] = annotated_from_python["hgnc_symbol"].map(sort_genes)
        annotated_from_R["hgnc_symbol"] = annotated_from_R["hgnc_symbol"].map(normalize_genes)
        annotated_from_R["hgnc_symbol"] = annotated_from_R["hgnc_symbol"].map(sort_genes)
        mask = annotated_from_python["annotatedJunction"] != annotated_from_R["annotatedJunction"]
        diff = pd.concat([
            annotated_from_python[mask].add_suffix("_python"),
            annotated_from_R[mask].add_suffix("_R")
        ], axis=1).sort_index()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(diff[[  'Start_python' , 'End_python', 'annotatedJunction_python', 'annotatedJunction_R']])
            print(diff.shape)
        #pd.testing.assert_frame_equal(annotated_from_python, annotated_from_R, check_dtype=False)
        output_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/predicted_means.csv", index_col=0)
        output_from_python = torch.sigmoid(torch.Tensor(df_out.values))
        output_from_python_df = pd.DataFrame(output_from_python.numpy().T, columns=df_out.index)
        pd.testing.assert_frame_equal(output_from_python_df, output_from_R.reset_index(drop=True), check_dtype=False)
        print('Calculating p-values of the results from R')
        pvals, _ = get_pvals(x_true=fraser_dataset.K.values.T,
                         res=fraser_dataset.N.values.T,
                         mu=output_from_R,
                         sigma=rho_from_R.squeeze(),
                         df0=None,
                         how='two-sided',
                         theta=None,
                         dis='bb',
                         n_jobs=config.n_jobs)
        pvals_from_R = pd.read_csv("/s/project/py_fraser/exported_counts/pVals.csv")
        pvals_df = pd.DataFrame(pvals).T
        pvals_df.columns = pvals_from_R.columns
        pd.testing.assert_frame_equal(pvals_df, pvals_from_R, check_dtype=False)


TestPipelineFRASER = TestPipelineFRASER()
TestPipelineFRASER.test_run("PROTRIDER/config.yaml")

