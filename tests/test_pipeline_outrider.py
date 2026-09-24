"""
Tests for OUTRIDER pipeline

This module tests basic pipeline execution with different input formats
and output saving options.
"""

import numpy as np
import pandas as pd
import tempfile
import warnings
from pathlib import Path
import torch
import pytest
from unittest.mock import patch
from scipy.special import gammaln

from protrider import ProtriderConfig, run
from protrider.pipeline import Result
from protrider.model import ModelInfo
from protrider.dispersions import OutriderDispersion


class TestPipelineOUTRIDER:
    """Test class for standard (non-CV) pipeline execution."""

    @pytest.mark.parametrize(
        ("analysis", "pseudocount", "expected_warnings"),
        [("outrider", 1.0, 0), ("outrider", 0.5, 1), ("protrider", 0.5, 0)],
    )
    def test_nonstandard_outrider_pseudocount_warns_at_run(
        self, analysis, pseudocount, expected_warnings, tmp_path
    ):
        config = ProtriderConfig(
            out_dir=str(tmp_path),
            input_intensities="counts.tsv",
            analysis=analysis,
            autoencoder_loss="NLL" if analysis == "outrider" else "MSE",
            pval_dist="nb" if analysis == "outrider" else "t",
            pseudocount=pseudocount,
        )
        with patch("protrider.pipeline._run_protrider_standard", return_value=(None, None)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                run(config)
        messages = [str(w.message) for w in caught if "original OUTRIDER documentation" in str(w.message)]
        assert len(messages) == expected_warnings
        if expected_warnings:
            assert "uses 1.0" in messages[0]

    def test_run_with_file_paths(self, gene_expression_path, gene_annotation_path):
        """Test running OUTRIDER with file paths in config."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                analysis='outrider',
                autoencoder_loss='NLL',
                pval_dist='nb',
                input_intensities=gene_expression_path,
                gtf=gene_annotation_path,
                index_col='geneID',
                seed=42,
                n_epochs=2,  # Short for testing
                gs_epochs=2,  # Short for testing
                find_q_method='5',  # Fixed q for speed
                verbose=False
            )

            result, model_info = run(config)

            # Check result type
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
            assert model_info.epochs_run == 2
            assert not model_info.stopped_early
            assert model_info.stopping_reason == "maximum epochs reached"

            # Check result contains expected dataframes
            assert isinstance(result.df_out, pd.DataFrame)
            assert isinstance(result.df_res, pd.DataFrame)
            assert isinstance(result.df_pvals, pd.DataFrame)
            assert isinstance(result.df_Z, pd.DataFrame)
            assert isinstance(result.df_pvals_adj, pd.DataFrame)

            # Check shapes are consistent
            n_samples, n_proteins = result.df_res.shape
            assert result.df_pvals.shape == (n_samples, n_proteins)
            assert result.df_Z.shape == (n_samples, n_proteins)

    def test_configured_pseudocount_controls_outrider_results(
        self, gene_expression_path, gene_annotation_path, tmp_path, caplog
    ):
        config = ProtriderConfig(
            out_dir=str(tmp_path),
            analysis="outrider",
            autoencoder_loss="NLL",
            pval_dist="nb",
            input_intensities=gene_expression_path,
            gtf=gene_annotation_path,
            index_col="geneID",
            find_q_method="5",
            n_epochs=1,
            device="cpu",
            pseudocount=0.5,
        )
        with pytest.warns(UserWarning, match="original OUTRIDER documentation uses 1.0"):
            result, _ = run(config)
        counts = result.dataset.raw_counts_filtered.to_numpy()
        expected = result.df_expected_counts.to_numpy()
        expected_log2fc = np.log2(counts + 0.5) - np.log2(expected + 0.5)
        np.testing.assert_allclose(result.log2fc, expected_log2fc, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result.fc, (counts + 0.5) / (expected + 0.5), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            result.dataset.normalized_log_counts,
            np.log((counts + 0.5) / result.dataset.size_factors_array),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.df_Z,
            (expected_log2fc - expected_log2fc.mean(axis=0))
            / expected_log2fc.std(axis=0, ddof=1),
            rtol=1e-5,
            atol=1e-5,
        )

        config.pseudocount = 0.75
        with caplog.at_level("WARNING"):
            with pytest.warns(UserWarning, match="original OUTRIDER documentation uses 1.0"):
                _, model_info = run(config)
        assert "checkpoint OUTRIDER model input does not match" in caplog.text
        assert model_info.epochs_run == 1

    @pytest.mark.parametrize("interval", [1, 10])
    def test_final_statistics_and_checkpoint_share_one_fit(
        self, gene_expression_path, gene_annotation_path, tmp_path, interval
    ):
        """The checkpoint and every reported statistic use the final theta fit."""
        config = ProtriderConfig(
            out_dir=str(tmp_path),
            analysis="outrider",
            autoencoder_loss="NLL",
            pval_dist="nb",
            input_intensities=gene_expression_path,
            gtf=gene_annotation_path,
            index_col="geneID",
            find_q_method="5",
            n_epochs=1,
            batch_size=3,
            outrider_precision="float64",
            outrider_theta_fit_interval=interval,
            device="cpu",
        )

        with patch.object(OutriderDispersion, "fit", autospec=True,
                          side_effect=OutriderDispersion.fit) as fit:
            result, model_info = run(config)
        assert fit.call_count == (1 // interval) + 2
        checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)
        theta = result.dispersions["theta"].to_numpy()

        np.testing.assert_allclose(checkpoint["theta"], theta)
        np.testing.assert_allclose(
            checkpoint["mean_scale"], result.mu["mu"].to_numpy()
        )

        counts = result.dataset.raw_counts_filtered.to_numpy()
        expected = result.df_expected_counts.to_numpy()
        theta_matrix = theta[None, :]
        log_prob = (
            gammaln(counts + theta_matrix)
            - gammaln(theta_matrix)
            - gammaln(counts + 1)
            + theta_matrix * np.log(theta_matrix / (theta_matrix + expected))
            + counts * np.log(expected / (theta_matrix + expected))
        )
        np.testing.assert_allclose(
            model_info.final_consistent_nll,
            -log_prob.mean(),
            rtol=1e-6,
        )

        reloaded_result, _ = run(config)
        np.testing.assert_allclose(
            reloaded_result.df_expected_counts,
            result.df_expected_counts,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            reloaded_result.dispersions,
            result.dispersions,
            rtol=1e-6,
        )

    def test_early_stopping_records_training_outcome(
        self, gene_expression_path, gene_annotation_path, tmp_path
    ):
        config = ProtriderConfig(
            out_dir=str(tmp_path),
            analysis="outrider",
            autoencoder_loss="NLL",
            pval_dist="nb",
            input_intensities=gene_expression_path,
            gtf=gene_annotation_path,
            index_col="geneID",
            find_q_method="5",
            n_epochs=4,
            device="cpu",
            outrider_early_stopping=True,
            outrider_early_stopping_patience=2,
            outrider_early_stopping_min_delta=1e6,
            outrider_early_stopping_min_epochs=3,
        )

        _, model_info = run(config)

        assert model_info.epochs_run == 3
        assert 1 <= model_info.best_epoch <= model_info.epochs_run
        assert model_info.stopped_early
        assert "full-cohort NLL did not improve" in model_info.stopping_reason.item()
        assert len(model_info.train_losses) == model_info.epochs_run

    def test_pca_only_run_saves_refitted_dispersion(
        self, gene_expression_path, gene_annotation_path, tmp_path
    ):
        config = ProtriderConfig(
            out_dir=str(tmp_path),
            analysis="outrider",
            autoencoder_loss="NLL",
            pval_dist="nb",
            input_intensities=gene_expression_path,
            gtf=gene_annotation_path,
            index_col="geneID",
            find_q_method="5",
            autoencoder_training=False,
            device="cpu",
        )

        result, _ = run(config)
        checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)
        np.testing.assert_allclose(
            checkpoint["theta"], result.dispersions["theta"].to_numpy()
        )
        np.testing.assert_allclose(
            checkpoint["mean_scale"], result.mu["mu"].to_numpy()
        )
