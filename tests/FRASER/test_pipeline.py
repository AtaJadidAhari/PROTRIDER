"""
Tests for the full FRASER pipeline (protrider.run(analysis='fraser')).

This module tests basic pipeline execution with different input formats
and output saving options.
"""
import tempfile
from pathlib import Path

import pandas as pd

from protrider import ProtriderConfig, run
from protrider.pipeline import Result
from protrider.model import ModelInfo

from conftest import (
    GTF_PATH,
    MIN_EXPRESSION_IN_ONE_SAMPLE,
    QUANTILE,
    QUANTILE_MIN_EXPRESSION,
    PSEUDOCOUNT,
    MIN_DELTA_PSI,
)


def _make_config(out_dir, split_reads_path, unsplit_reads_path, **overrides):
    kwargs = dict(
        out_dir=out_dir,
        analysis="fraser",
        input_intensities="unused",  # required by ProtriderConfig, ignored by FraserDataset
        split_reads=split_reads_path,
        unsplit_reads=unsplit_reads_path,
        gtf=str(GTF_PATH),
        min_expression_in_one_sample=MIN_EXPRESSION_IN_ONE_SAMPLE,
        quantile_for_filtering=QUANTILE,
        quantile_min_expression=QUANTILE_MIN_EXPRESSION,
        pseudocount=PSEUDOCOUNT,
        min_delta_psi=MIN_DELTA_PSI,
        find_q_method="2",  # fixed q for speed
        n_epochs=2,
        gs_epochs=2,
        autoencoder_loss="BBL",  # FRASER's Beta-Binomial reconstruction loss
        pval_dist="bb",
        pval_adj="holm",
        device="cpu",
        n_jobs=1,
        verbose=False,
    )
    kwargs.update(overrides)
    return ProtriderConfig(**kwargs)


class TestPipelineFraser:
    """Test class for standard (non-CV) FRASER pipeline execution."""

    def test_run_with_file_paths(self, split_reads_path, unsplit_reads_path):
        """Result exposes all expected assays after a run."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir, split_reads_path, unsplit_reads_path)
            result, model_info = run(config)

            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)

            assert isinstance(result.df_out, pd.DataFrame)
            assert isinstance(result.df_res, pd.DataFrame)  # delta psi (jaccard) after run()
            assert isinstance(result.df_pvals, pd.DataFrame)
            assert isinstance(result.df_pvals_adj, pd.DataFrame)
            assert isinstance(result.mu, pd.DataFrame)  # predicted means
            assert isinstance(result.dispersions, pd.DataFrame)
            assert result.gene_level_info is not None

            n_samples, n_junctions = result.df_res.shape
            assert result.df_pvals.shape == (n_samples, n_junctions)
            assert result.dispersions.shape[0] == n_junctions

    def test_save_results_long_format(self, split_reads_path, unsplit_reads_path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir, split_reads_path, unsplit_reads_path)
            result, _ = run(config)

            result.save(tmp_dir, format="long", analysis="fraser")

            out_dir = Path(tmp_dir)
            junc_all = out_dir / "fraser_summary_all_junctions.csv.gz"
            junc_filtered = out_dir / "fraser_summary_filtered_junctions.csv"
            gene_all = out_dir / "fraser_summary_all_genes.csv"
            gene_filtered = out_dir / "fraser_summary_filtered_genes.csv"
            for p in (junc_all, junc_filtered, gene_all, gene_filtered):
                assert p.exists(), f"expected output file missing: {p}"

            junc_df = pd.read_csv(junc_all)
            expected_cols = {"sampleID", "JUNCTION_PSI", "JUNCTION_DELTAPSI",
                              "JUNCTION_PVALUE", "JUNCTION_PADJ", "JUNCTION_outlier"}
            assert expected_cols.issubset(junc_df.columns)

            gene_df = pd.read_csv(gene_all)
            expected_gene_cols = {"sampleID", "gene_id", "GENE_PVALUE", "GENE_PADJ", "GENE_outlier"}
            assert expected_gene_cols.issubset(gene_df.columns)
