import gc
import gzip
import logging
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .config import ProtriderConfig
from .datasets import (OmicDataset, ProtriderDataset,
                       ProtriderKfoldCVGenerator, ProtriderLOOCVGenerator,
                       ProtriderSubset)
from .model import (ModelInfo, MSEBCELoss, OmicAutoencoder, find_latent_dim,
                    init_model, train, train_val)
from .plots import plot_cv_loss
from .stats import adjust_pvals, fit_residuals, get_pvals

__all__ = ["run"]

logger = logging.getLogger(__name__)


def save_model(model: OmicAutoencoder, checkpoint_path: str, q: int) -> None:
    """Save model state dict and metadata to checkpoint path.
    
    Args:
        model: Trained OmicAutoencoder model
        checkpoint_path: Path where to save the model checkpoint
        q: Latent dimension
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'q': q,
        'n_layers': model.n_layers,
        'presence_absence': model.presence_absence,
    }, checkpoint_path)
    
    logger.info(f'Saved model to {checkpoint_path}')


def load_model(dataset: Union[ProtriderDataset, ProtriderSubset], checkpoint_path: str, 
               config: ProtriderConfig) -> Tuple[Optional[OmicAutoencoder], Optional[int]]:
    """Load model from checkpoint path if it exists.
    
    Args:
        dataset: Dataset used for model initialization
        checkpoint_path: Path to the model checkpoint file
        config: ProtriderConfig object
        
    Returns:
        Tuple of (model, q) if model exists and loads successfully, (None, None) otherwise
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.info(f'No existing model found at {checkpoint_path}')
        return None, None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=config.device_torch, weights_only=False)
        q = checkpoint['q']
        n_layers = checkpoint['n_layers']
        presence_absence = checkpoint.get('presence_absence', False)
        
        logger.info(f'Loading model from {checkpoint_path} (q={q}, n_layers={n_layers})')
        
        # Initialize model with saved architecture
        n_cov = dataset.covariates.shape[1]
        n_prots = dataset.X.shape[1]
        model = OmicAutoencoder(
            in_dim=n_prots, 
            latent_dim=q, 
            n_layers=n_layers, 
            h_dim=config.h_dim, 
            n_cov=n_cov,
            prot_means=None,
            presence_absence=presence_absence
        )
        model.double().to(config.device_torch)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info('Successfully loaded model')
        return model, q
        
    except Exception as e:
        logger.warning(f'Failed to load model from {checkpoint_path}: {e}')
        return None, None


def log_peak_memory(label: str):
    """Log the process' peak resident set size (maxrss) so far, for tracking down memory spikes."""
    maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info('[mem] %s: peak RSS so far = %.2f GB', label, maxrss_kb / 1e6)


class StepTimer:
    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0

    def step(self, name):
        now = time.perf_counter()
        logger.info('%s: %.2fs (total %.2fs)', name, now - self._last, now - self._t0)
        self._last = now


@dataclass
class Result:
    """Stores results from a standard run of PROTRIDER."""
    #dataset: ProtriderDataset
    dataset: OmicDataset
    df_out: pd.DataFrame
    df_res: pd.DataFrame
    df_pvals: pd.DataFrame
    df_pvals_one_sided: pd.DataFrame
    df_presence: pd.DataFrame
    df_Z: pd.DataFrame
    df_pvals_adj: pd.DataFrame
    log2fc: np.ndarray
    fc: np.ndarray
    latent_values: pd.DataFrame
    pval_dist: str = 'gaussian'  # Distribution used for p-value computation
    outlier_threshold: float = 0.1  # Threshold for determining outliers (adjusted p value cutoff)
    # OUTRIDER results
    dispersions: pd.DataFrame = None # Dispersions for OUTRIDER or FRASER
    mu: pd.DataFrame = None # OUTRIDER mu
    # FRASER results
    delta_psi_cutoff: float = 0.1  # Threshold for delta PSI (jaccard) for fraser
    min_count: int = 5  # Minimum total count threshold for fraser
    gene_level_info: Optional[dict] = None  # Gene-level information for Fraser 
    junction_chunk_rows:int = 500_000  # Target number of long-format rows (samples * junctions) held in memory per chunk.

    def detect_outliers(self, df_res, prefix = 'PROTEIN', analysis: str = "protrider") -> pd.DataFrame:
        if analysis == "fraser":
            df_res[f'{prefix}_outlier'] = (df_res[f'{prefix}_PADJ'] <= self.outlier_threshold) & (df_res['JUNCTION_DELTAPSI'].abs() >= self.delta_psi_cutoff) & (df_res['totalCounts'] >= self.min_count)
        else:
            df_res[f'{prefix}_outlier'] = df_res[f'{prefix}_PADJ'].apply(lambda x:  x <= self.outlier_threshold)
            outs_per_sample = df_res.groupby('sampleID')[f'{prefix}_outlier'].sum().values
            n_out_median = np.nanmedian(outs_per_sample)
            n_out_max = np.nanmax(outs_per_sample)
            n_out_total = np.nansum(outs_per_sample)
            logger.info(f'Finished computing pvalues. No. outliers per sample in median: {n_out_median}, max: {n_out_max}, total: {n_out_total}')
            logger.info(f'Finished computing pvalues. No. outliers per sample in median: {np.nanmedian(outs_per_sample)}')
            df_res[f'{prefix}_outlier'] = df_res[f'{prefix}_PADJ'].apply(lambda x:  x <= self.outlier_threshold)
        return df_res

    def write_fraser_gene_level_table(self, all_path: str, filtered_path: str) -> pd.DataFrame:
        pvals_for_by = self.gene_level_info['pvals_for_by']       # genes x samples
        kept_junctions = self.gene_level_info['kept_junctions']   # genes x samples
        gene_padj = self.df_pvals_adj                             # genes x samples

        junction_wide = {
            'JUNCTION_PSI': self.dataset.jaccard_index.T,
            'JUNCTION_DELTAPSI': self.df_res,
            'JUNCTION_PVALUE': self.df_pvals,
            'JUNCTION_PADJ': self.gene_level_info['junction_pvals_adj'].T,
            **self.dataset.calculate_counts(),
        }  # samples x junctions

        junc_static_meta = pd.concat([
            self.dataset.split_reads[["seqnames", "start", "end", "width", "strand"]],
            self.dataset.intron_ranges[["annotatedJunction"]],
            ], axis=1)

        gene_ids = pvals_for_by.index
        n_samples = len(self.dataset.sample_ids)
        chunk_size = max(1, self.junction_chunk_rows // max(1, n_samples))

        gene_id_to_name = self.dataset.build_gene_id_to_name_map(self.dataset.gtf) if self.dataset.gtf is not None else {}

        def build_chunk(gene_chunk):
            gene_junctions = kept_junctions.reindex(gene_chunk)

            # genes x samples -> one row per gene/sample
            gene_level = pd.concat({
                'GENE_PVALUE': pvals_for_by.reindex(gene_chunk),
                'GENE_PADJ': gene_padj.reindex(gene_chunk),
                'junctionID': gene_junctions,
            }, axis=1).stack(future_stack=True).reset_index()
            gene_level.columns = ['gene_id', 'sampleID', 'GENE_PVALUE', 'GENE_PADJ', 'junctionID']
            gene_level['hgnc_symbol'] = gene_level['gene_id'].map(gene_id_to_name).fillna(gene_level['gene_id'])

            junc_ids_needed = pd.unique(gene_junctions.to_numpy().ravel())
            junc_ids_needed = junc_ids_needed[pd.notna(junc_ids_needed)]

            # samples x junctions -> looked up by (sampleID, junctionID) for the representative junction
            junction_level = pd.concat({k: v.reindex(columns=junc_ids_needed) for k, v in junction_wide.items()}, axis=1).stack(future_stack=True).reset_index()
            junction_level = junction_level.rename(columns={junction_level.columns[0]: 'sampleID', junction_level.columns[1]: 'junctionID'})

            df_chunk = gene_level.merge(junction_level, on=['sampleID', 'junctionID'], how='left')
            del gene_level, junction_level

            df_chunk = df_chunk.merge(junc_static_meta.reindex(junc_ids_needed), left_on='junctionID', right_index=True, how='left')
            df_chunk.drop(columns=['junctionID'], inplace=True, errors='ignore')
            return df_chunk

        return self.stream_write_long_table(all_path, filtered_path, gene_ids, chunk_size, build_chunk, prefix='GENE', analysis='fraser', compressed=False)

    def fraser_junc_meta_with_aggregates(self) -> pd.DataFrame:
        gene_id_to_name = self.dataset.build_gene_id_to_name_map(self.dataset.gtf) if self.dataset.gtf is not None else {}

        def _map_gene_ids(gene_id_str):
            if pd.isna(gene_id_str):
                return pd.NA
            return ";".join(sorted(set(gene_id_to_name.get(gid, gid) for gid in gene_id_str.split(';'))))

        hgnc_symbol = self.dataset.intron_ranges["gene_id"].apply(_map_gene_ids)
        junc_meta = pd.concat([
            self.dataset.split_reads[["seqnames", "start", "end", "width", "strand"]],
            pd.DataFrame({'hgnc_symbol': hgnc_symbol, 'annotatedJunction': self.dataset.intron_ranges['annotatedJunction']}),
            ], axis=1)
        # for col in ["seqnames", "strand", "hgnc_symbol", "annotatedJunction"]:
        #     junc_meta[col] = junc_meta[col].astype("category")

        n_samples = len(self.dataset.sample_ids)
        has_gene = junc_meta['hgnc_symbol'].notna()
        junc_meta['numSamplesPerJunc'] = n_samples
        junc_meta['numSamplesPerGene'] = np.where(has_gene, n_samples, np.nan)
        junc_meta['numEventsPerGene'] = junc_meta.groupby('hgnc_symbol', observed=True)['hgnc_symbol'].transform('size')
        junc_meta.loc[~has_gene, 'numEventsPerGene'] = np.nan
        return junc_meta

    def stream_write_long_table(self, all_path: str, filtered_path: str, chunk_ids, chunk_size: int,
                                  build_chunk, prefix: str, analysis: str, compressed: bool = True) -> pd.DataFrame:
        open_fn = gzip.open if compressed else open
        filtered_chunks = []
        total_rows = 0
        with open_fn(all_path, 'wt') as f_out:
            for i in range(0, len(chunk_ids), chunk_size):
                id_chunk = chunk_ids[i:i + chunk_size]
                df_chunk = build_chunk(id_chunk)
                df_chunk = self.finalize_long(df_chunk, prefix=prefix, analysis=analysis)

                first = (i == 0)
                df_chunk.to_csv(f_out, header=first, index=False)
                total_rows += len(df_chunk)
                df_filtered_chunk = df_chunk[df_chunk[f'{prefix}_outlier']].copy()
                logger.debug(f'Wrote {prefix.lower()}-level output summary chunk {i}-{i + len(id_chunk)} to {all_path}')
                filtered_chunks.append(df_filtered_chunk)

                del df_chunk, df_filtered_chunk
                gc.collect()
            logger.info(f'Finished writing {prefix.lower()}-level output summary to {all_path} with total rows: {total_rows}')

        df_filtered = pd.concat(filtered_chunks, ignore_index=True).sort_values(f'{prefix}_PVALUE')
        df_filtered.to_csv(filtered_path, index=False)
        logger.info(f'\t--- Removing non-significant {prefix.lower()}s. \n\tOriginal len: {total_rows}, new len: {df_filtered.shape[0]}---')
        logger.info(f'Saved {prefix.lower()}-level filtered output summary with shape {df_filtered.shape} to {filtered_path}')
        return df_filtered

    def write_fraser_junction_level_table(self, all_path: str, filtered_path: str) -> pd.DataFrame:
        wide = {
            'JUNCTION_PSI': self.dataset.jaccard_index.T.astype(np.float32),
            'JUNCTION_DELTAPSI': self.df_res.astype(np.float32),
            'JUNCTION_PVALUE': self.df_pvals.astype(np.float32),
            'JUNCTION_PADJ': self.gene_level_info['junction_pvals_adj'].T.astype(np.float32),
            **self.dataset.calculate_counts(),
        }
        junc_meta = self.fraser_junc_meta_with_aggregates()

        junction_ids = wide['JUNCTION_PSI'].columns
        for v in wide.values():
            junction_ids = junction_ids.union(v.columns, sort=False)
        n_samples = len(self.dataset.sample_ids)
        chunk_size = max(1, self.junction_chunk_rows // max(1, n_samples))

        def build_chunk(junc_chunk):
            combined = pd.concat({k: v.reindex(columns=junc_chunk) for k, v in wide.items()}, axis=1)
            df_chunk = combined.stack(future_stack=True).reset_index()
            df_chunk.columns = ['sampleID', 'junctionID'] + list(combined.columns.get_level_values(0).unique())
            del combined

            df_chunk = df_chunk.merge(junc_meta.reindex(junc_chunk), left_on='junctionID', right_index=True, how='left')
            df_chunk.drop(columns=['junctionID'], inplace=True, errors='ignore')
            return df_chunk

        return self.stream_write_long_table(all_path, filtered_path, junction_ids, chunk_size, build_chunk, prefix='JUNCTION', analysis='fraser')

    def finalize_long(self, df_res: pd.DataFrame, prefix: str, analysis: str) -> pd.DataFrame:
        df_res['pvalDistribution'] = self.pval_dist
        df_res = self.detect_outliers(df_res, prefix=prefix, analysis=analysis)
        return df_res.sort_values(by=f'{prefix}_PVALUE', ascending=True)

    def save(self, out_dir: str, format: Literal["wide", "long"] = "wide",
            include_all: bool = False, analysis: str = "protrider") -> Optional[pd.DataFrame]:
        """
        Save result dataframes to CSV files.

        Args:
            out_dir: Output directory path where files will be saved
            format: Output format - "wide" saves separate CSV files for each metric,
                   "long" saves a single combined CSV in long format
            include_all: If False, only save significant results in long format
                        (uses self.outlier_threshold for filtering). Ignored when
                        analysis == "fraser": all 4 combinations of
                        include_all/aggregate are always written.

        Returns:
            DataFrame if format="long", None if format="wide"

        """
        summary_dir = out_dir
        out_dir = f"{out_dir}/{analysis}"
        os.makedirs(out_dir, exist_ok=True)
        if format == "wide":
            logger.info('=== Saving results in wide format ===')
            
            # AE input
            out_p = f'{out_dir}/processed_input.csv'
            self.dataset.data.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved processed_input to {out_p}")

            # AE output
            out_p = f'{out_dir}/output.csv'
            self.df_out.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved output to {out_p}")

            # residuals
            out_p = f'{out_dir}/residuals.csv'
            self.df_res.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved residuals to {out_p}")

            # presence probabilities
            if self.df_presence is not None:
                out_p = f'{out_dir}/presence_probs.csv'
                self.df_presence.T.to_csv(out_p, header=True, index=True)
                logger.info(f"Saved presence probabilities to {out_p}")

            # p-values
            out_p = f'{out_dir}/pvals.csv'
            self.df_pvals.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved P-values to {out_p}")

            # left-sided p-values
            if self.df_pvals_one_sided is not None:
                out_p = f'{out_dir}/pvals_one_sided.csv'
                self.df_pvals_one_sided.T.to_csv(out_p, header=True, index=True)
                logger.info(f"Saved left-sided P-values to {out_p}")

            # p-values adj
            out_p = f'{out_dir}/pvals_adj.csv'
            self.df_pvals_adj.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved adjusted P-values to {out_p}")

            # Z-scores
            out_p = f'{out_dir}/zscores.csv'
            self.df_Z.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved z scores to {out_p}")

            # log2fc
            out_p = f'{out_dir}/log2fc.csv'
            self.log2fc.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved log2fc scores to {out_p}")

            # fc
            out_p = f'{out_dir}/fc.csv'
            self.fc.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved fc scores to {out_p}")

            # latent vector
            out_p = f'{out_dir}/latent_values.csv'
            self.latent_values.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved latent values to {out_p}")
            
            # AE raw, filtered input
            if analysis == 'outrider':
                out_p = f'{out_dir}/raw_filtered_input.csv'
                self.dataset.raw_filtered.T.to_csv(out_p, header=True, index=True)
                logger.info(f"Saved raw_filtered_input to {out_p}")

                # sizefactors
                sf_df = pd.DataFrame(self.dataset.size_factors.cpu(), index=self.dataset.raw_filtered.index, columns=["size_factor"])
                out_p = f'{out_dir}/sizefactors.csv'
                sf_df.to_csv(out_p, header=True, index=True)
                logger.info(f"Saved sizefactors to to {out_p}")
                
                # dispersions
                out_p = f'{out_dir}/theta.csv'
                self.dispersions.to_csv(out_p, header=True, index=True)
                logger.info(f"Saved thetas to to {out_p}")

                # mu
                out_p = f'{out_dir}/mu.csv'
                self.mu.to_csv(out_p, header=True, index=True)
                logger.info(f"Saved mus to to {out_p}")

            return None
            
        elif format == "long":
            logger.info('=== Saving results in long format ===')

            if analysis == 'fraser':
                #log_peak_memory('before building junction-level table')
                all_p = f"{summary_dir}/{analysis}_summary_all_junctions.csv.gz"
                filtered_p = f"{summary_dir}/{analysis}_summary_filtered_junctions.csv"
                df_junc_filtered = self.write_fraser_junction_level_table(all_p, filtered_p)
                #log_peak_memory('after building junction-level table')

                all_p = f"{summary_dir}/{analysis}_summary_all_genes.csv"
                filtered_p = f"{summary_dir}/{analysis}_summary_filtered_genes.csv"
                df_gene_filtered = self.write_fraser_gene_level_table(all_p, filtered_p)
                #log_peak_memory('after building gene-level table')

                result = df_junc_filtered
                del df_gene_filtered
                gc.collect()
                #log_peak_memory('after freeing gene long table')

                return result

            if self.df_pvals.shape != self.df_pvals_adj.shape:
                self.df_pvals_adj = self.df_pvals_adj.T

            dfs_to_melt = {
                'PROTEIN_LOG2INT': self.dataset.data,
                'PROTEIN_EXPECTED_LOG2INT': self.df_out,
                'PROTEIN_INT': self.dataset.raw_data,
                'PROTEIN_ZSCORE': self.df_Z,
                'PROTEIN_PVALUE': self.df_pvals,
                'PROTEIN_PADJ': self.df_pvals_adj,
                'PROTEIN_LOG2FC': self.log2fc,
                'PROTEIN_FC': self.fc,
            }
            if self.df_presence is not None:
                dfs_to_melt['pred_presence_probability'] = self.df_presence

            combined = pd.concat(dfs_to_melt, axis=1)
            df_res = combined.stack(future_stack=True).reset_index()
            df_res.columns = ['sampleID', 'proteinID'] + list(dfs_to_melt.keys())

            df_res = self.finalize_long(df_res, prefix='PROTEIN', analysis=analysis)

            if not include_all:
                original_len = df_res.shape[0]
                df_res = df_res.query('PROTEIN_outlier==True')
                logger.info(
                    f'\t--- Removing non-significant sample-protein combinations. \n\tOriginal len: {original_len}, new len: {df_res.shape[0]}---')

            out_p = f"{summary_dir}/{analysis}_summary.csv"
            df_res.to_csv(out_p, index=None)
            logger.info(f'Saved output summary with shape {df_res.shape} to {out_p}')

            return df_res
    
    def plot_pvals(self, out_dir: str = None, **kwargs):
        """
        Plot p-value distributions.
        
        Args:
            out_dir: Optional output directory for saving plots. If None, plots are returned but not saved.
            **kwargs: Additional arguments passed to the plotting function (plot_title, fontsize, distribution)
            
        Returns:
            tuple: (histogram_plot, qq_plot) - plotnine plot objects
            
        Example:
            >>> hist, qq = result.plot_pvals()  # Interactive use
            >>> hist.draw()
            >>> result.plot_pvals(out_dir='output/')  # Save plots
        """
        from . import plots

        # Pass the one-sided p-values DataFrame
        pvals_one_sided = self.df_pvals_one_sided if hasattr(self, 'df_pvals_one_sided') else None
        return plots.plot_pvals(
            output_dir=out_dir, 
            pvals_one_sided=pvals_one_sided,
            distribution=self.pval_dist,
            **kwargs
        )
    
    def plot_aberrant_per_sample(self, out_dir: str = None, **kwargs):
        """
        Plot number of aberrant proteins per sample.
        
        Args:
            out_dir: Optional output directory for saving plots. If None, plot is returned but not saved.
            **kwargs: Additional arguments passed to the plotting function (plot_title, fontsize)
            
        Returns:
            plotnine plot object
            
        Example:
            >>> plot = result.plot_aberrant_per_sample()  # Interactive use
            >>> plot.draw()
            >>> result.plot_aberrant_per_sample(out_dir='output/')  # Save plot
        """
        from . import plots

        # Build protrider_summary DataFrame on the fly
        ae_out = self.df_out
        ae_in = self.dataset.data
        raw_in = self.dataset.raw_data
        zscores = self.df_Z
        pvals = self.df_pvals
        pvals_adj = self.df_pvals_adj
        log2fc = self.log2fc
        fc = self.fc
        presence = self.df_presence

        ae_out = (ae_out.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_EXPECTED_LOG2INT'}))
        ae_in = (ae_in.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2INT'}))
        raw_in = (raw_in.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_INT'}))
        zscores = (zscores.reset_index().melt(id_vars='sampleID')
                   .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_ZSCORE'}))
        pvals = (pvals.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PVALUE'}))
        pvals_adj = (pvals_adj.reset_index().melt(id_vars='sampleID')
                     .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PADJ'}))
        log2fc = (log2fc.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2FC'}))
        fc = (fc.reset_index().melt(id_vars='sampleID')
              .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_FC'}))

        merge_cols = ['sampleID', 'proteinID']
        protrider_summary = (ae_in.merge(ae_out, on=merge_cols)
                  .merge(raw_in, on=merge_cols)
                  .merge(zscores, on=merge_cols)
                  .merge(pvals, on=merge_cols)
                  .merge(pvals_adj, on=merge_cols)
                  .merge(log2fc, on=merge_cols)
                  .merge(fc, on=merge_cols)
                  ).reset_index(drop=True)

        if presence is not None:
            presence = (presence.reset_index().melt(id_vars='sampleID')
                        .rename(columns={'variable': 'proteinID', 'value': 'pred_presence_probability'}))
            protrider_summary = protrider_summary.merge(presence, on=merge_cols).reset_index(drop=True)

        protrider_summary['PROTEIN_outlier'] = protrider_summary['PROTEIN_PADJ'].apply(
            lambda x: x <= self.outlier_threshold)
        protrider_summary['pvalDistribution'] = self.pval_dist
        
        return plots.plot_aberrant_per_sample(
            output_dir=out_dir,
            protrider_summary=protrider_summary,
            **kwargs
        )
    
    def plot_expected_vs_observed(self, protein_id: str, out_dir: str = None, **kwargs):
        """
        Plot expected vs observed intensities for a specific protein.
        
        Args:
            protein_id: Protein identifier to plot
            out_dir: Optional output directory for saving plots. If None, plot is returned but not saved.
            **kwargs: Additional arguments passed to the plotting function (plot_title, fontsize)
            
        Returns:
            plotnine plot object
            
        Example:
            >>> plot = result.plot_expected_vs_observed('PROT123')  # Interactive use
            >>> plot.draw()
            >>> result.plot_expected_vs_observed('PROT123', out_dir='output/')  # Save plot
        """
        from . import plots

        # Prepare data for plotting - transpose to match expected format
        processed_input = self.dataset.data.T.reset_index()
        processed_input.columns.name = None
        processed_input = processed_input.rename(columns={'sampleID': 'proteinID'})
        
        output_data = self.df_out.T.reset_index()
        output_data.columns.name = None
        output_data = output_data.rename(columns={'sampleID': 'proteinID'})
        
        # Build protrider_summary on the fly (same as plot_aberrant_per_sample)
        ae_out = self.df_out
        ae_in = self.dataset.data
        raw_in = self.dataset.raw_data
        zscores = self.df_Z
        pvals = self.df_pvals
        pvals_adj = self.df_pvals_adj
        log2fc = self.log2fc
        fc = self.fc
        presence = self.df_presence

        ae_out = (ae_out.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_EXPECTED_LOG2INT'}))
        ae_in = (ae_in.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2INT'}))
        raw_in = (raw_in.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_INT'}))
        zscores = (zscores.reset_index().melt(id_vars='sampleID')
                   .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_ZSCORE'}))
        pvals = (pvals.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PVALUE'}))
        pvals_adj = (pvals_adj.reset_index().melt(id_vars='sampleID')
                     .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PADJ'}))
        log2fc = (log2fc.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2FC'}))
        fc = (fc.reset_index().melt(id_vars='sampleID')
              .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_FC'}))

        merge_cols = ['sampleID', 'proteinID']
        protrider_summary = (ae_in.merge(ae_out, on=merge_cols)
                  .merge(raw_in, on=merge_cols)
                  .merge(zscores, on=merge_cols)
                  .merge(pvals, on=merge_cols)
                  .merge(pvals_adj, on=merge_cols)
                  .merge(log2fc, on=merge_cols)
                  .merge(fc, on=merge_cols)
                  ).reset_index(drop=True)

        if presence is not None:
            presence = (presence.reset_index().melt(id_vars='sampleID')
                        .rename(columns={'variable': 'proteinID', 'value': 'pred_presence_probability'}))
            protrider_summary = protrider_summary.merge(presence, on=merge_cols).reset_index(drop=True)

        protrider_summary['PROTEIN_outlier'] = protrider_summary['PROTEIN_PADJ'].apply(
            lambda x: x <= self.outlier_threshold)
        protrider_summary['pvalDistribution'] = self.pval_dist
        
        return plots.plot_expected_vs_observed(
            protein_id=protein_id,
            output_dir=out_dir,
            processed_input=processed_input,
            output_data=output_data,
            protrider_summary=protrider_summary,
            **kwargs
        )


def run(config: ProtriderConfig) -> Tuple[Result, ModelInfo]:
    """
    Run PROTRIDER protein outlier detection.
    
    Automatically dispatches to cross-validation or standard mode based on config.cross_val.
    All inputs including data files are specified in the config.
    
    Args:
        config: ProtriderConfig object with all configuration parameters including:
                - input_intensities: File path (str)
                  * File format: columns = samples, rows = proteins
                - sample_annotation: File path (str) or None
                  * Format: rows = samples

    Returns:
        Tuple of (Result, ModelInfo)
        - Result: Contains all output dataframes (residuals, p-values, z-scores, etc.)
        - ModelInfo: Contains model metadata (q, learning_rate, losses, etc.)
                    For CV runs, includes df_folds with fold assignments
    
    Examples:
        >>> # Using file paths (file format: columns = samples, rows = proteins)
        >>> config = ProtriderConfig(
        ...     out_dir='output',
        ...     input_intensities='data.csv',  # Columns = samples
        ...     sample_annotation='annotations.csv',
        ...     cross_val=False
        ... )
        >>> result, model_info = run(config)
        
        >>> # With cross-validation
        >>> config_cv = ProtriderConfig(
        ...     out_dir='output',
        ...     input_intensities='data.csv',
        ...     cross_val=True,
        ...     n_folds=5
        ... )
        >>> result, model_info = run(config_cv)
    """
    if config.seed is not None:
        logger.info("Setting random seed: %s", config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    if config.cross_val:
        logger.info('Running PROTRIDER with cross-validation')
        return _run_protrider_cv(config, config.input_intensities, config.sample_annotation)
    else:
        logger.info('Running PROTRIDER in standard mode')
        return _run_protrider_standard(config, config.input_intensities, config.sample_annotation)


def _run_protrider_standard(
    config: ProtriderConfig,
    input_intensities: Union[str, pd.DataFrame],
    sample_annotation: Union[str, pd.DataFrame, None]
) -> Tuple[Result, ModelInfo]:
    """
    Perform protein outlier detection in a single run (internal function).
    
    Args:
        config: ProtriderConfig object with all configuration parameters
        input_intensities: Protein intensities as file path or pandas DataFrame
                          - File: columns = samples, rows = proteins
                          - DataFrame: rows = samples, columns = proteins
        sample_annotation: Sample annotations as file path, DataFrame, or None
                          - Format: rows = samples

    Returns:
        Tuple of (Result, ModelInfo)
    """
    timer = StepTimer()


    # 1. Initialize dataset
    logger.info('Initializing dataset')
    dataset  = OmicDataset(analysis=config.analysis,
                           split_reads=config.split_reads,
                           unsplit_reads=config.unsplit_reads,
                            minExpressionInOneSample = config.min_expression_in_one_sample, 
                            quantile = config.quantile_for_filtering,
                            quantileMinExpression = config.quantile_min_expression,
                            min_delta_psi=config.min_delta_psi,
                            pseudocount = config.pseudocount,
                            input_intensities=input_intensities,
                            index_col=config.index_col,
                            sa_file=sample_annotation,
                            cov_used=config.cov_used,
                            log_func=config.log_func,
                            maxNA_filter=config.max_allowed_NAs_per_protein,
                            fpkm_cutoff=config.fpkmCutoff,
                            gtf=config.gtf,
                            device=config.device_torch,
                            input_format=config.input_format)

    # 1.5 Determine checkpoint path and try to load existing model
    model = None
    q = None
    # Use custom checkpoint path if specified, otherwise default to out_dir/model.pt
    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
    else:
        checkpoint_path = Path(config.out_dir) / 'model.pt'

    timer.step('Initializing dataset')
    # 2. Find latent dim
    logger.info('Finding latent dimension')
    q = find_latent_dim(dataset, method=config.find_q_method,
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
                        config=config,
                        )

    logger.info(
        f'Latent dimension found with method {config.find_q_method}: {q}')
    timer.step('Finding latent dimension')
    # 3. Init model with found latent dim
    model = init_model(dataset, q,
                       init_wPCA=config.init_pca,
                       n_layer=config.n_layers,
                       h_dim=config.h_dim,
                       device=config.device_torch,
                       presence_absence=config.presence_absence if config.n_layers == 1 else False,
                       model_type=config.analysis
                       )
    
    criterion = model.set_loss(autoencoder_loss = config.autoencoder_loss, lambda_presence_absence = config.lambda_presence_absence) 
    logger.info('Model:\n%s', model)
    logger.info('Device: %s', config.device_torch)
    timer.step('Initializing model')

 ## 4. Compute initial loss
    df_out, theta, df_presence, init_loss, init_mse_loss, init_bce_loss = _inference(dataset, model, criterion, batch_size=config.batch_size)
    logger.info('Initial loss after model init: %s, mse loss: %s, bce loss: %s', init_loss, init_mse_loss,
                init_bce_loss)
    timer.step('Computing initial loss')

    final_loss = 10**4
    train_losses = []
    if config.autoencoder_training:
        logger.info('Fitting model')
        _, _, _, train_losses = train(dataset, model, criterion, n_epochs=config.n_epochs, learning_rate=float(config.lr), batch_size=config.batch_size)
        timer.step('Fitting model')

        df_out, theta, df_presence, final_loss, final_reconstruction_loss, final_bce_loss = _inference(dataset, model, criterion, batch_size=config.batch_size)
        logger.info('Final loss: %s, mse loss: %s, bce loss: %s', final_loss, final_reconstruction_loss, final_bce_loss)
        timer.step('Computing final loss')
        save_model(model, str(checkpoint_path), q)
        timer.step('Saving model')
    else:
        final_loss = init_loss
        timer.step('Model fitting skipped')

    # 6. Compute residuals, pvals, zscores
    logger.info('Computing statistics')
    model_input = model if config.analysis != "protrider" else None
    mu, sigma, df0, df_res = fit_residuals(dataset, df_out, model_input, config) # for fraser df_res is N.T
    latent_values = model.get_latent_values()
    timer.step('Fitting residuals')
    x_true = dataset.K.T.values if config.analysis == "fraser" else dataset.raw_filtered.values
    pvals, Z = get_pvals(x_true=x_true,
                         res=df_res.values,
                         mu=mu,
                         sigma=sigma,
                         df0=df0,
                         how=config.pval_sided,
                         theta=theta,
                         dis=config.pval_dist,
                         n_jobs=config.n_jobs)
    pvals_one_sided = None
    if config.calculate_one_sided_pval:
        pvals_one_sided, _ = get_pvals(x_true=x_true,
                                    res=df_res.values,
                                    mu=mu,
                                    sigma=sigma,
                                    df0=df0,
                                    how='left',
                                    theta=theta,
                                    dis=config.pval_dist,
                                    n_jobs=config.n_jobs)
    timer.step('Computing p-values')
    group_ids = dataset.intron_ranges["gene_id"] if config.analysis == "fraser" else None
    pvals_adj, gene_level_info = adjust_pvals(pvals, method=config.pval_adj, group_ids=group_ids, aggregate=True, n_jobs=config.n_jobs,
                                              index=dataset.data.columns if config.analysis == "fraser" else dataset.data.index,
                                              columns=dataset.data.index if config.analysis == "fraser" else dataset.data.columns,
                                              transpose=config.analysis == "fraser")
    timer.step('Adjusting p-values')
    #  df_res for Fraser from here on is the actual delta psi
    df_res = np.round(dataset.jaccard_index.T - mu.T, decimals=2) if config.analysis == "fraser" else df_res

    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence,
                             pvals=pvals, Z=Z, pvals_one_sided=pvals_one_sided, pvals_adj=pvals_adj,
                             pseudocount=config.pseudocount, outlier_threshold=config.outlier_threshold,
                             delta_psi_cutoff = config.delta_psi_threshold, min_count=config.min_count,
                             base_fn=config.base_fn, pval_dist=config.pval_dist,
                             gene_level_info=gene_level_info, latent_values=latent_values,
                             dispersions=theta, mu=mu)
    model_info = ModelInfo(q=np.array(q), learning_rate=np.array(config.lr),
                           n_epochs=np.array(config.n_epochs), test_loss=np.array(final_loss),
                           train_losses=np.array(train_losses), df_folds=None)
    timer.step('Finalizing model')
    return result, model_info


def _run_protrider_cv(
    config: ProtriderConfig,
    input_intensities: Union[str, pd.DataFrame],
    sample_annotation: Union[str, pd.DataFrame, None]
) -> Tuple[Result, ModelInfo]:
    """
    Perform protein outlier detection with cross-validation (internal function).
    
    Args:
        config: ProtriderConfig object with all configuration parameters
        input_intensities: Protein intensities as file path or pandas DataFrame
                          - File: columns = samples, rows = proteins
                          - DataFrame: rows = samples, columns = proteins
        sample_annotation: Sample annotations as file path, DataFrame, or None
                          - Format: rows = samples

    Returns:
        Tuple of (Result, ModelInfo) - ModelInfo.df_folds contains fold assignments for CV
    """
    # If fit_every_fold is set to True, the model will estimate the residual distribution parameters on every fold
    # using the train-val set.
    # If set to False, the model will estimate the residual distribution parameters on the final test residuals
    fit_every_fold = config.fit_every_fold

    # 1. Initialize cross validation generator
    logger.info('Initializing cross validation')
    if config.n_folds is not None:
        cv_gen = ProtriderKfoldCVGenerator(input_intensities, sample_annotation, config.index_col,
                                           config.cov_used, config.max_allowed_NAs_per_protein, config.log_func,
                                           num_folds=config.n_folds, device=config.device_torch,
                                           input_format=config.input_format)
    else:
        cv_gen = ProtriderLOOCVGenerator(input_intensities, sample_annotation, config.index_col, config.cov_used,
                                         config.max_allowed_NAs_per_protein, config.log_func, device=config.device_torch,
                                         input_format=config.input_format)
    dataset = cv_gen.dataset
    criterion = MSEBCELoss(
        presence_absence=config.presence_absence, lambda_bce=config.lambda_presence_absence)
    # test results
    pvals_list = []
    Z_list = []
    df_out_list = []
    df_res_list = []
    df_presence_list = []
    test_loss_list = []
    train_losses_list = []
    q_list = []
    df0_list = []
    folds_list = []
    # 2. Loop over folds
    for fold, (train_subset, val_subset, test_subset) in enumerate(cv_gen):

        logger.info(f'Fold {fold}')
        logger.info(f'Train subset size: {len(train_subset)}')
        logger.info(f'Validation subset size: {len(val_subset)}')
        logger.info(f'Test subset size: {len(test_subset)}')

        # 3. Find latent dim
        logger.info('Finding latent dimension')
        pca_subset = ProtriderSubset.concat([train_subset, val_subset])
        q = find_latent_dim(pca_subset, method=config.find_q_method,
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
                            n_jobs=config.n_jobs,
                            config=config,
                            )
        logger.info(
            f'Latent dimension found with method {config.find_q_method}: {q}')

        # 4. Init model with found latent dim
        model = init_model(train_subset, q, init_wPCA=config.init_pca, n_layer=config.n_layers,
                           h_dim=config.h_dim, device=config.device_torch, presence_absence=config.presence_absence)

        logger.info('Model:\n%s', model)
        logger.info('Device: %s', config.device_torch)

        # 5. Compute initial MSE loss
        df_out_train, theta_train, df_presence_train, train_loss, train_mse_loss, train_bce_loss = _inference(train_subset, model,
                                                                                                 criterion)
        df_out_val, theta_val, df_presence_val, val_loss, val_mse_loss, val_bce_loss = _inference(
            val_subset, model, criterion)
        logger.info(f'Train loss after model init: {train_loss}')
        logger.info(f'Validation loss after model init: {val_loss}')
        if config.autoencoder_training:
            logger.info('Fitting model')
            # 6. Train model
            # todo train validate (hyperparameter tuning)
            # todo pass validation set as well
            train_losses, val_losses = train_val(train_subset, val_subset, model, criterion,
                                                 n_epochs=config.n_epochs,
                                                 learning_rate=float(config.lr),
                                                 batch_size=config.batch_size,
                                                 patience=config.early_stopping_patience,
                                                 min_delta=config.early_stopping_min_delta)
            plot_cv_loss(train_losses, val_losses, fold, config.out_dir)
            train_losses_list.append(train_losses)
        else:
            train_losses_list.append([])

        df_out_train, theta_train, df_presence_train, train_loss, train_mse_loss, train_bce_loss = _inference(train_subset, model,
                                                                                                 criterion)
        df_out_val, theta_val, df_presence_val, val_loss, val_mse_loss, val_bce_loss = _inference(
            val_subset, model, criterion)
        logger.info(f'Fold {fold} train loss: {train_loss}')
        logger.info(f'Fold {fold} validation loss: {val_loss}')

        # 7. Compute residuals on test set
        logger.info('Running model on test set')
        df_out_test, theta_test, df_presence_test, test_loss, test_mse_loss, test_bce_loss = _inference(test_subset, model,
                                                                                            criterion)
        logger.info(f'Fold {fold} test loss: {test_loss}')
        df_res_test = test_subset.data - df_out_test  # log data - pred data

        df_out_list.append(df_out_test)
        if df_presence_test is not None:
            df_presence_list.append(df_presence_test)
        df_res_list.append(df_res_test)
        test_loss_list.append(test_loss)
        q_list.append(q)
        folds_list.extend([fold] * len(df_out_test))

        if fit_every_fold:
            # 8. Fit residual distribution on train-val set
            logger.info(
                'Estimating residual distribution parameters on train-val set')
            #df_res_val = val_subset.data - df_out_val  # log data - pred data
            #df_res_train = train_subset.data - df_out_train  # log data - pred data

            train_val_subset = ProtriderSubset.concat([val_subset, train_subset])
            mu, sigma, df0, df_res = fit_residuals(train_val_subset,
                                                   pd.concat([df_out_val, df_out_train]),
                                                   None, config)
    
            #mu, sigma, df0 = fit_residuals(
            #    pd.concat([df_res_train, df_res_val]).values, dis=config.pval_dist, n_jobs=config.n_jobs)
            pvals, Z = get_pvals(x_true=dataset.raw_filtered.values, res=df_res_test.values, mu=mu,
                                 sigma=sigma, df0=df0, how=config.pval_sided, n_jobs=config.n_jobs)
            pvals_list.append(pvals)
            Z_list.append(Z)
            df0_list.append(df0)

    df_res = pd.concat(df_res_list)
    df_out = pd.concat(df_out_list)
    df_presence = pd.concat(
        df_presence_list) if df_presence_list != [] else None
    df_folds = pd.DataFrame({'fold': folds_list, }, index=df_out.index)

    # Compute p-values on test set
    if fit_every_fold:
        pvals = np.concatenate(pvals_list)
        Z = np.concatenate(Z_list)
    else:
        logger.info('Estimating residual distribution parameters')
        mu, sigma, df0, df_res = fit_residuals(dataset, df_out, None, config)
        #mu, sigma, df0 = fit_residuals(df_res.values, dis=config.pval_dist, n_jobs=config.n_jobs)
        pvals, Z = get_pvals(x_true=dataset.raw_filtered.values, res=df_res.values, mu=mu, sigma=sigma, df0=df0,
                             how=config.pval_sided, n_jobs=config.n_jobs)
        # Repeat df0 for each sample in the output
        df0_list = [df0] * len(df_out)

    # Compute one-sided p-values (used for some plots)
    pvals_one_sided = None
    if config.calculate_one_sided_pval:
        pvals_one_sided, _ = get_pvals(x_true=dataset.raw_filtered.values,
                                    res=df_res.values,
                                    mu=mu,
                                    sigma=sigma,
                                    df0=df0 if config.pval_dist == 't' else None,
                                    how='left',
                                    dis=config.pval_dist,
                                    n_jobs=config.n_jobs)

    group_ids = dataset.intron_ranges["gene_id"] if config.analysis == "fraser" else None
    pvals_adj, gene_level_info = adjust_pvals(pvals, method=config.pval_adj, group_ids=group_ids, aggregate=True, n_jobs=config.n_jobs,
                                              index=dataset.data.columns if config.analysis == "fraser" else dataset.data.index,
                                              columns=dataset.data.index if config.analysis == "fraser" else dataset.data.columns,
                                              transpose=config.analysis == "fraser")
    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence,
                             pvals=pvals, Z=Z, pvals_one_sided=pvals_one_sided, pvals_adj=pvals_adj,
                             pseudocount=config.pseudocount, outlier_threshold=config.outlier_threshold,
                             delta_psi_cutoff = config.delta_psi_threshold, min_count=config.min_count,
                             base_fn=config.base_fn, pval_dist=config.pval_dist,
                             gene_level_info=gene_level_info)
    model_info = ModelInfo(q=np.array(q_list), learning_rate=np.array(config.lr),
                           n_epochs=np.array(config.n_epochs), test_loss=np.array(test_loss_list),
                           train_losses=np.array(train_losses_list, dtype=object), 
                           df0=np.array(df0_list), df_folds=df_folds)
    return result, model_info


def _inference(dataset: Union[ProtriderDataset, ProtriderSubset], model: OmicAutoencoder, criterion: MSEBCELoss, batch_size=None, use_cpu=True):
    
    # Save original device
    orig_device = next(model.parameters()).device

    # Extract tensors
    X = dataset.X
    mask = dataset.torch_mask
    cov = dataset.covariates
    raw_x = getattr(dataset, "raw_x", None)
    K_tensor = getattr(dataset, "K_tensor", None)
    N_tensor = getattr(dataset, "N_tensor", None)

    # Move tensors to appropriate device (CPU or original GPU)
    device = "cpu" if use_cpu else orig_device

    # Move model to CPU if requested
    model = model.to(device)
    if model.encoder.omic_means is not None:
        model.encoder.omic_means = model.encoder.omic_means.to(device)
    X = X.to(device)
    mask = mask.to(device)
    if cov is not None:
        cov = cov.to(device)
    if raw_x is not None:
        raw_x = raw_x.to(device)
    if K_tensor is not None:
        K_tensor = K_tensor.to(device)
    if N_tensor is not None:
        N_tensor = N_tensor.to(device)

    model.eval()
    with torch.no_grad():

        if model.model_type == "protrider":
            X_out = model(X, mask, cond=cov)
            loss, reconstruction_loss, bce_loss = criterion(
                X_out, X, mask, detached=True
            )
            theta = None

        elif model.model_type == "outrider":
            X_out = model(X, mask, cond=cov)

            _, theta = model.get_dispersion_parameters()
            loss, reconstruction_loss, bce_loss = criterion(
                (theta, torch.exp(X_out) * torch.tensor(dataset.size_factors, device=device)),
                raw_x,
                detached=True
            )
        elif model.model_type == "fraser":
            X_out = model(X, mask, cond=cov)
            _, rho = model.get_dispersion_parameters()

            loss, reconstruction_loss, bce_loss = criterion(
                (rho, torch.sigmoid(X_out)),
                (K_tensor, N_tensor),
                detached=True
            )
            theta = rho 

        # Presence/Absence extension
        df_presence = None
        if model.presence_absence:
            presence_hat = torch.sigmoid(X_out[1])
            X_out = X_out[0]

            df_presence = pd.DataFrame(
                presence_hat.detach().cpu().numpy(),
                index=dataset.data.index,
                columns=dataset.data.columns
            )

        # Output intensities
        df_out = pd.DataFrame(
            X_out.detach().cpu().numpy(),
            index=dataset.data.index,
            columns=dataset.data.columns
        )

    # Restore model back to original device
    if use_cpu:
        model = model.to(orig_device)
        if model.encoder.omic_means is not None:
            model.encoder.omic_means = model.encoder.omic_means.to(orig_device)

    return df_out, theta, df_presence, loss, reconstruction_loss, bce_loss


def _format_results(df_out, df_res, df_presence, pvals, Z, pvals_one_sided, pvals_adj, dataset,
                    pseudocount, outlier_threshold, base_fn, pval_dist, delta_psi_cutoff=0.1, min_count=5,
                    gene_level_info = None, latent_values = None, dispersions=None, mu=None):
    # Store as df
    if not isinstance(pvals_adj, pd.DataFrame):
        df_pvals_adj = pd.DataFrame(pvals_adj)
        df_pvals_adj.columns = dataset.data.columns
        df_pvals_adj.index = dataset.data.index
    else: 
        df_pvals_adj = pvals_adj

    # Store as df
    if not isinstance(pvals, pd.DataFrame):
        df_pvals = pd.DataFrame(pvals)
        df_pvals.columns = dataset.data.columns
        df_pvals.index = dataset.data.index
    else:
        df_pvals = pvals

    # Store as df
    if not isinstance(Z, pd.DataFrame) and Z is not None:
        df_Z = pd.DataFrame(Z)
        df_Z.columns = dataset.data.columns
        df_Z.index = dataset.data.index
    else:
        df_Z = Z

    if not isinstance(pvals_one_sided, pd.DataFrame) and pvals_one_sided is not None: 
        df_pvals_one_sided = pd.DataFrame(pvals_one_sided)
        df_pvals_one_sided.columns = dataset.data.columns
        df_pvals_one_sided.index = dataset.data.index
    else:
        df_pvals_one_sided = pvals_one_sided

    pseudocount = pseudocount  # 0.01
    if base_fn is not None:
        log2fc = np.log2(base_fn(dataset.data) + pseudocount) - \
            np.log2(base_fn(df_out) + pseudocount)
        fc = (base_fn(dataset.data) + pseudocount) / \
            (base_fn(df_out) + pseudocount)
    else:
        log2fc, fc = None, None

    if dispersions is not None:
        dispersions = pd.DataFrame(dispersions)
        dispersions.columns = ["theta"]
        dispersions.index = dataset.data.columns
    if mu is not None:
        mu = np.asarray(mu)
        if mu.ndim == 1:
            # OUTRIDER/PROTRIDER: single dispersion-scale value per feature
            mu = pd.DataFrame(mu, columns=["mu"], index=dataset.data.columns)
        else:
            # FRASER: predicted probability per junction (rows) x sample (cols)
            mu = pd.DataFrame(mu, index=dataset.data.columns, columns=dataset.data.index)

    df_latent_values = pd.DataFrame(latent_values)
    df_latent_values.index = dataset.data.index
    df_latent_values = df_latent_values.T
    df_latent_values.index.name = "enc_dim"
        

    return Result(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence, df_pvals=df_pvals, df_Z=df_Z,
                  df_pvals_one_sided=df_pvals_one_sided, df_pvals_adj=df_pvals_adj, log2fc=log2fc, fc=fc,
                  pval_dist=pval_dist, outlier_threshold=outlier_threshold, delta_psi_cutoff = delta_psi_cutoff, min_count = min_count,
                  gene_level_info = gene_level_info, latent_values=df_latent_values, dispersions=dispersions, mu=mu)
