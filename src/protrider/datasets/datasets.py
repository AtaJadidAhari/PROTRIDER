from __future__ import annotations

from typing import Iterable, Optional, Callable
from warnings import filters
#from dpath import values
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import copy
from abc import ABC
from optht import optht
import logging
from .covariates import parse_covariates
from .protein_intensities import read, preprocess_protein_intensities
from pydeseq2.preprocessing import deseq2_norm
import pandas as pd
import re
#from pathlib import Path
#import torch.nn.functional as F
import time
import pyranges as pr



logger = logging.getLogger(__name__)

class PCADataset(ABC):
    def __init__(self):
        # centered log data around the protein means
        self.centered_log_data_noNA = None
        self.U = None
        self.s = None
        self.Vt = None

    def perform_svd(self):
        self.U, self.s, self.Vt = np.linalg.svd(self.centered_log_data_noNA, full_matrices=False)
        logger.info(f'Finished fitting SVD with shapes U: {self.U.shape}, s: {self.s.shape}, Vt: {self.Vt.shape}')

    def find_enc_dim_optht(self):
        if self.s is None:
            self.perform_svd()

        try:
            q = optht(self.centered_log_data_noNA, sv=self.s, sigma=None) 
        except ValueError as e:
            q = 1

        if q < 2:
            logger.warning(f"Optimal latent space dimension is smaller than 2. Check your count matrix and"
                             "verify that all samples have the expected number of counts\nFor now, the latent space dimension is set to 2.")
            q = 2
        return q


class ProtriderDataset(Dataset, PCADataset):
    def __init__(self, input_intensities: str, index_col: str, 
                 sa_file: Optional[str] = None,
                 cov_used: Optional[list] = None, log_func: Callable = np.log,
                 maxNA_filter: float = 0.3, device: torch.device = torch.device('cpu'),
                 input_format: str = "proteins_as_rows"):
        """Initialize ProtriderDataset.
        
        Args:
            input_intensities: Path to protein intensities file (CSV, TSV, or Parquet)
            index_col: Name of the index column containing protein IDs
            sa_file: Path to sample annotations file (CSV/TSV), or None
                    - Format: rows = samples
            cov_used: List of covariate column names to use, or None
            log_func: Function to apply log transformation (default: np.log)
            maxNA_filter: Maximum allowed proportion of NA values (default: 0.3)
            device: PyTorch device (default: 'cpu')
            input_format: Format of input file:
                         - "proteins_as_rows": proteins are rows, samples are columns (default)
                         - "proteins_as_columns": samples are rows, proteins are columns
        """
        super().__init__()
        self.device = device

        # Read and preprocess protein intensities
        unfiltered_data = read(input_intensities, index_col, input_format)
        self.data, self.raw_data, self.size_factors = preprocess_protein_intensities(
            unfiltered_data, log_func, maxNA_filter
        )

        # Read and preprocess covariates
        if sa_file is not None and cov_used is not None:
            try:
                self.covariates, self.centered_covariates_noNA = parse_covariates(sa_file, cov_used, self.data.index)
                self.covariates = torch.from_numpy(self.covariates)
                self.centered_covariates_noNA = torch.from_numpy(self.centered_covariates_noNA)
            except ValueError:
                logger.warning("No valid covariates found after parsing.")
                self.covariates = torch.empty(self.data.shape[0], 0)
                self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)
        else:
            self.covariates = torch.empty(self.data.shape[0], 0)
            self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)

        
        # store protein means
        self.omic_means = np.nanmean(self.data, axis=0, keepdims=1)

        ## Center and mask NaNs in input
        self.mask = ~np.isfinite(self.data)
        self.centered_log_data_noNA = self.data - self.omic_means
        self.centered_log_data_noNA = np.where(self.mask, 0, self.centered_log_data_noNA)

        # Input and output of autoencoder is:
        # uncentered data without NaNs, replacing NANs with means
        self.X = self.centered_log_data_noNA + self.omic_means  ## same as data but without NAs

        ## to torch
        self.X = torch.tensor(self.X)
        # self.X_target = self.X ### needed for outlier injection
        self.mask = np.array(self.mask.values)
        self.torch_mask = torch.tensor(self.mask)
        self.omic_means_torch = torch.from_numpy(self.omic_means).squeeze(0)

        ### Send data to cpu/gpu device
        self.X = self.X.to(device)
        self.torch_mask = self.torch_mask.to(device)
        self.covariates = self.covariates.to(device)
        self.omic_means_torch = self.omic_means_torch.to(device)
        # self.presence = (~self.torch_mask).long()
        self.raw_filtered = pd.DataFrame()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.torch_mask[idx], self.covariates[idx], self.omic_means_torch)


class ProtriderSubset(Subset, PCADataset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.omic_means = np.nanmean(self.data, axis=0, keepdims=1)
        self.omic_means_torch = torch.from_numpy(self.omic_means).squeeze(0).to(dataset.device)

    @property
    def X(self):
        return self.dataset.X[self.indices]

    @property
    def data(self):
        return self.dataset.data.iloc[self.indices]

    @property
    def raw_data(self):
        return self.dataset.raw_data.iloc[self.indices]

    @property
    def mask(self):
        return self.dataset.mask[self.indices]

    @property
    def torch_mask(self):
        return self.dataset.torch_mask[self.indices]

    @property
    def centered_log_data_noNA(self):
        return self.dataset.centered_log_data_noNA[self.indices]

    @property
    def covariates(self):
        return self.dataset.covariates[self.indices]

    @staticmethod
    def concat(subsets: Iterable['ProtriderSubset']):
        """
        Concatenate multiple ProtriderSubset instances into a single one.
        """
        indices = np.concatenate([subset.indices for subset in subsets])
        return ProtriderSubset(subsets[0].dataset, indices)

    def deepcopy_to_dataset(self) -> Dataset:
        """
        Convert the ProtriderSubset instance back to a ProtriderDataset instance.
        """
        dataset = copy.deepcopy(self.dataset)
        dataset.X = self.X
        dataset.data = self.data
        dataset.raw_data = self.raw_data
        dataset.mask = self.mask
        dataset.torch_mask = self.torch_mask
        dataset.centered_log_data_noNA = self.centered_log_data_noNA
        dataset.covariates = self.covariates
        dataset.omic_means = self.omic_means
        dataset.omic_means_torch = self.omic_means_torch
        return dataset


class OutriderDataset(Dataset, PCADataset):
    def __init__(self, input_intensities: str, index_col: str,
                 sa_file: Optional[str] = None,
                 cov_used: Optional[list] = None, log_func: Callable = np.log,
                 fpkm_cutoff: int = 1, gtf: str = None,
                 device: torch.device = torch.device('cpu'),
                 input_format: str = "proteins_as_rows"):
        super().__init__()

        # Read and preprocess protein intensities
        self.data = read(input_intensities, index_col)
        # self.data, self.raw_data, self.size_factors = preprocess_protein_intensities(
        #     unfiltered_data, log_func, maxNA_filter
        # )

        self.device = device
        self.data.index.names = ['sampleID']
        self.data.columns.name = 'proteinID'
        self.gtf = gtf
        logger.info(f'Finished reading raw data with shape: {self.data.shape}')

        self.raw_data = copy.deepcopy(self.data)  ## for storing output

        # normalize data
        _, size_factors = deseq2_norm(self.data.replace(np.nan, 0))
        self.size_factors = size_factors[:, np.newaxis]


        # filter based on fpkm
        logger.info(f'Calculating fpkms')

        gene_lengths_df = self.gene_exonic_lengths_from_gtf(self.gtf)
        self.fpkms = self.calculate_fpkm(self.raw_data.T, gene_lengths_df)

        logger.info(f'Filtering based on fpkm')
        _, self.passed_filter = self.filter_genes_by_fpkm(self.fpkms.T, fpkm_cutoff=fpkm_cutoff, percentage=0.05)
        self.data = self.data.loc[:, self.passed_filter.values]

        self.raw_filtered = copy.deepcopy(self.data)  ## for storing output/plotting
        # normalize with size_factors
        self.data = np.log((1 + self.data) / self.size_factors)


        #### FINISHED PREPROCESSING

        # store protein means
        self.omic_means = np.nanmean(self.data, axis=0, keepdims=1)

        ## Center and mask NaNs in input
        self.mask = ~np.isfinite(self.data)
        self.centered_log_data_noNA = self.data - self.omic_means
        self.centered_log_data_noNA = np.where(self.mask, 0, self.centered_log_data_noNA)

        omic_means_df = pd.DataFrame({"xbar": np.ravel(self.omic_means), "geneID": self.data.columns})
        # Input and output of autoencoder is:
        # uncentered data without NaNs, replacing NANs with means
        # self.X = self.centered_log_data_noNA + self.omic_means  ## same as data but without NAs
        self.X = self.centered_log_data_noNA

        ## to torch
        self.X = torch.tensor(self.X)
        self.raw_x = torch.tensor(self.raw_filtered.values)

        self.mask = np.array(self.mask.values)
        self.torch_mask = torch.tensor(self.mask)
        self.omic_means_torch = torch.from_numpy(self.omic_means).squeeze(0)

        # Read and preprocess covariates
        if sa_file is not None and cov_used is not None:
            try:
                self.covariates, self.centered_covariates_noNA = parse_covariates(sa_file, cov_used, self.data.index)
                self.covariates = torch.from_numpy(self.covariates)
                self.centered_covariates_noNA = torch.from_numpy(self.centered_covariates_noNA)
            except ValueError as e:
                print(e)
                logger.warning("No valid covariates found after parsing.")
                self.covariates = torch.empty(self.data.shape[0], 0)
                self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)
        else:
            self.covariates = torch.empty(self.data.shape[0], 0)
            self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)

        ### Send data to cpu/gpu device
        self.X = self.X.to(device)
        self.torch_mask = self.torch_mask.to(device)
        self.covariates = self.covariates.to(device)
        self.omic_means_torch = self.omic_means_torch.to(device)
        self.raw_x = self.raw_x.to(device)
        self.size_factors = torch.tensor(self.size_factors).to(device)
        # self.presence = (~self.torch_mask).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.torch_mask[idx], self.covariates[idx], self.omic_means_torch, self.raw_x[idx], self.size_factors[idx])

    def filter_genes_by_fpkm(self, fpkm_matrix, fpkm_cutoff=1, percentage=0.05):
        """
        Filters genes based on minimum FPKM in at least 5% of samples.

        Parameters:
            fpkm_matrix (pd.DataFrame): samples × genes (rows are samples, columns are genes)
            min_fpkm (float): Minimum FPKM threshold
            percentage (float): Fraction of samples that must pass the threshold

        Returns:
            passed_filter (pd.Series): Boolean mask for each gene (index = gene names)
            filtered_matrix (pd.DataFrame): Matrix after filtering (samples × genes)
        """
        # Number of samples (rows)
        n_samples = fpkm_matrix.shape[0]

        # Minimum number of samples a gene must pass the threshold in
        min_samples = max(1.0, percentage * n_samples)

        # Boolean mask: for each gene (column), count how many samples (rows) have FPKM ≥ min_fpkm
        passed_filter = (fpkm_matrix > fpkm_cutoff).sum(axis=0) >= min_samples

        logger.info(f"{len(passed_filter) - passed_filter.sum()} genes out of {len(passed_filter)} are filtered out. New shape: ({n_samples}, {passed_filter.sum()})")

        # Apply filter to keep only columns (genes) that passed
        filtered_matrix = fpkm_matrix.loc[:, passed_filter]

        return filtered_matrix, passed_filter

    def gene_exonic_lengths_from_gtf(self, gtf_path: str) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: gene_id, exonic_length
        Collapses overlapping exons per gene across all transcripts --> not necessarily 
        mane_transcript gene length
        """
        def extract_gene_id(attr):
            m = re.search(r'gene_id\s+"([^"]+)"', attr)
            if m:
                return m.group(1)
            m = re.search(r'gene_id=([^;]+)', attr)
            if m:
                return m.group(1)
            return None


        def merge_and_sum(intervals: np.ndarray) -> int:
            """
            intervals: Nx2 numpy array of [start, end] (inclusive, 1-based)
            returns total union length in bp.
            """
            if intervals.size == 0:
                return 0

            # sort by start then end
            intervals = intervals[np.lexsort((intervals[:,1], intervals[:,0]))]

            total = 0
            cur_start, cur_end = intervals[0]

            for s, e in intervals[1:]:
                if s <= cur_end + 1:
                    if e > cur_end:
                        cur_end = e
                else:
                    total += (cur_end - cur_start + 1)
                    cur_start, cur_end = s, e

            total += (cur_end - cur_start + 1)
            return int(total)

        # define col names of gtf file
        cols = ["seqname","source","feature","start","end","score","strand","frame","attribute"]
        gtf = pd.read_csv(
            gtf_path, sep="\t", comment="#", header=None, names=cols,
            dtype={"seqname":"category","source":"category","feature":"category",
                   "start":int,"end":int,"score":"string","strand":"category",
                   "frame":"string","attribute":"string"}
        )

        # keep only exons
        exons = gtf[gtf["feature"] == "exon"].copy()
        if exons.empty:
            return pd.DataFrame(columns=["gene_id","exonic_length"])

        # gene_id
        exons["gene_id"] = exons["attribute"].map(extract_gene_id)
        exons = exons.dropna(subset=["gene_id"])

        # for safety, ensure start <= end
        bad = (exons["end"] < exons["start"])
        if bad.any():
            exons.loc[bad, ["start","end"]] = exons.loc[bad, ["end","start"]].values

        # group by gene and merge intervals
        def _sum_for_gene(g):
            ivals = g[["start","end"]].to_numpy(dtype=np.int64)
            return merge_and_sum(ivals)

        lengths = (
            exons.groupby("gene_id", sort=False, observed=True)
                 .apply(_sum_for_gene)
                 .reset_index(name="exonic_length")
        )
        return lengths

    def calculate_fpkm(self, expr_df, lengths_df, robust=True):
        """
        expr_df: DataFrame, rows=genes, cols=samples, values=raw counts
        lengths_df: DataFrame with columns ["gene_id", "gene_length"] (bp)

        Returns: FPKM DataFrame with same shape as expr_df
        if robus = True, returns size-factor normalized fpkms, same as DESEQ2
        """

        lengths = lengths_df.set_index("gene_id")["exonic_length"]
        expr_df = expr_df.copy()

        # make sure order matches
        expr_df = expr_df.loc[expr_df.index.intersection(lengths.index)]
        lengths = lengths.loc[expr_df.index]

        if robust is False:
            library_size = expr_df.sum(axis=0)
        else:
            library_size = self.size_factors.T * np.exp(np.mean(np.log(expr_df.sum(axis=0))))

        fpkm = expr_df.div(lengths, axis=0) * 1e9
        fpkm = fpkm.div(library_size, axis=1)

        return fpkm

class OmicDataset(Dataset):
    def __new__(cls, analysis, *args, **kwargs):
        if cls is OmicDataset:   # Only auto-redirect when calling the base class
            if analysis == "fraser":
                return FraserDataset(*args, **kwargs)
            elif analysis == "outrider":
                return OutriderDataset(*args, **kwargs)
            elif analysis == "protrider":
                return ProtriderDataset(*args, **kwargs)
            else:
                raise ValueError(f"Unknown analysis type: {analysis}")
        return super().__new__(cls)
    #def __init__(self, analysis):
    #    self.analysis = analysis  # needed?
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass

class FraserDataset(Dataset, PCADataset): # inherit omic?
    def __init__(self, split_reads: str, unsplit_reads: str, sa_file: Optional[str] = None,
                 cov_used: Optional[list] = None):
        # read split and unsplit reads
        print("Reading split and unsplit reads")
        self.split_reads = read(input_intensities=split_reads, input_format = 'introns_as_rows') # ASK: index col??
        self.unsplit_reads = read(input_intensities=unsplit_reads, input_format = 'introns_as_rows')  # ASK: index col??
        self.samples_cols = self.split_reads.columns.intersection(self.unsplit_reads.columns) 
        

        print("Calculating K and N")
        start = time.time()
        # calculate K and N
        self.K = self.calculate_K()
        self.N = self.calculate_N() 
    
        duration = time.time() - start
        print(f"K and N calculated in {duration:.2f} seconds.")

        # calculate Jaccard index
        print("Calculating Jaccard index")
        start = time.time()
        self.jaccard_index = self.K / self.N
        
        # Jaccard filter expression
        self.filter_expression_jaccard()
        self.jaccard_index = self.jaccard_index.loc[self.passed_expression, :]
        duration = time.time() - start
        print(f"Jaccard index calculated and filtered in {duration:.2f} seconds.")

        # create X and center it
        self.data = self.create_data()
    
        self.data =  pd.DataFrame(self.data, index=self.samples_cols, columns=self.split_reads.index)
        
        self.X = torch.tensor(self.data.to_numpy(), dtype=torch.float64) - torch.nanmean(torch.tensor(self.data.to_numpy(), dtype=torch.float64), dim=0, keepdim=True)

        self.mask = ~np.isfinite(self.data)
        self.mask = np.array(self.mask.values)
        self.X = torch.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        self.torch_mask = torch.tensor(self.mask)
        self.centered_log_data_noNA = self.X.cpu().numpy()
        self.omic_means = np.nanmean(self.data, axis=0, keepdims=1) 
        self.omic_means_torch = torch.from_numpy(self.omic_means).squeeze(0)
        self.X = torch.tensor(self.centered_log_data_noNA + self.omic_means, dtype=torch.float64) #TEST
        self.raw_filtered = copy.deepcopy(self.data)
        self.raw_x = torch.tensor(self.raw_filtered.values)
        self.intron_ranges = pd.DataFrame({
            "Chromosome": self.split_reads["seqnames"],
            "StartId": self.split_reads["startID"],
            "EndId": self.split_reads["endID"],
            "Start": self.split_reads["start"],
            "End": self.split_reads["end"],
            "Strand": self.split_reads["strand"],
        })

        # Input and output of autoencoder is:
        # uncentered data without NaNs, replacing NANs with means
        #self.X = self.centered_log_data_noNA + self.junc_means  # ASK
        # Read and preprocess covariates
        if sa_file is not None and cov_used is not None:
            try:
                self.covariates, self.centered_covariates_noNA = parse_covariates(sa_file, cov_used, self.data.index)
                self.covariates = torch.from_numpy(self.covariates)
                self.centered_covariates_noNA = torch.from_numpy(self.centered_covariates_noNA)
            except ValueError:
                logger.warning("No valid covariates found after parsing.")
                self.covariates = torch.empty(self.data.shape[0], 0)
                self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)
        else:
            self.covariates = torch.empty(self.data.shape[0], 0)
            self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)

    def __getitem__(self, idx):
        return (self.X[idx], self.torch_mask[idx], self.covariates[idx], self.omic_means_torch, self.raw_x[idx])

    
    def calculate_K(self):
        return self.split_reads[self.samples_cols]
    
    def calculate_N(self):
        junc_idx = self.split_reads.index

        # Vectorized lookup: group unsplit reads by spliceSiteID, sum counts per site
        nsr = self.unsplit_reads.groupby("spliceSiteID")[self.samples_cols].sum()
        nsr_donor   = nsr.reindex(self.split_reads["startID"].values).fillna(0).astype(int).set_axis(junc_idx)
        nsr_acceptor = nsr.reindex(self.split_reads["endID"].values).fillna(0).astype(int).set_axis(junc_idx)
        self.nonsplitCounts = nsr_donor + nsr_acceptor  # junctions × samples

        # Vectorized lookup: group split reads by startID and endID, sum counts per site
        split_by_start = self.split_reads.groupby("startID")[self.samples_cols].sum()
        split_by_end   = self.split_reads.groupby("endID")[self.samples_cols].sum()
        alt_donor   = split_by_start.reindex(self.split_reads["startID"].values).fillna(0).astype(int).set_axis(junc_idx)
        alt_acceptor = split_by_end.reindex(self.split_reads["endID"].values).fillna(0).astype(int).set_axis(junc_idx)

        result_df = alt_donor + alt_acceptor + self.nonsplitCounts
        return result_df - self.K

    def filter_expression_jaccard(self, minExpressionInOneSample=20, quantile=0.95, quantileMinExpression=10):
        # Assure NP array
        cts = np.asarray(self.K)
        ctsN = np.asarray(self.N)

        # Compute cutoffs
        max_count = cts.max(axis=1)
        quantile_value_n = np.quantile(ctsN, quantile, axis=1)

        # Expression filter
        passed_expression = (
            (max_count >= minExpressionInOneSample) &
            (quantile_value_n >= quantileMinExpression)
        )
        self.passed_expression = passed_expression
    
    def create_data(self, pseudocount= 0.1):
        X = (self.K + pseudocount) / (self.N + 2 * pseudocount)
        X_logit = np.log(X / (1 - X))
        X_logit = X_logit.T
        return X_logit
        

    def get_site_index(self, psi_type: str = "jaccard") -> np.ndarray:
        """
        Returns a 1D array of splice site IDs (one per junction), matching R's getSiteIndex.
        Junctions sharing the same site ID are grouped for Holm FWER correction.

        psi_type:
            "jaccard" — each junction is its own group (no grouping, Holm is a no-op)
            "psi5"    — group by donor site: startID on + strand, endID on - strand
            "psi3"    — group by acceptor site: endID on + strand, startID on - strand
        """
        if psi_type == "jaccard":
            return np.arange(len(self.intron_ranges))

        start_id = self.intron_ranges["StartId"].to_numpy()
        end_id   = self.intron_ranges["EndId"].to_numpy()
        strand   = self.intron_ranges["Strand"].fillna("+").replace({"*": "+"}).to_numpy()

        if psi_type == "psi5":
            return np.where(strand == "+", start_id, end_id)
        elif psi_type == "psi3":
            return np.where(strand == "+", end_id, start_id)
        else:
            raise ValueError(f"Unknown psi_type '{psi_type}'. Choose from 'jaccard', 'psi5', 'psi3'.")

    def annotate_junctions(self, gtf_file: str, feature_name: str = "hgnc_symbol"):
        """Annotate junctions with gene symbols and reference overlap status.

        Adds two columns to self.intron_ranges:
          - feature_name (default 'hgnc_symbol'): semicolon-joined gene symbols from
            overlapping genes in the GTF; NA if no overlap.
          - 'annotatedJunction': 'both' (exact intron match), 'start' (only 5' end
            matches), 'end' (only 3' end matches), or 'none'.

        Parameters:
        gtf_file : str
            Path to a GTF annotation file.
        feature_name : str
            Column name for the gene symbol annotation. Default: 'hgnc_symbol'.
        """
        gtf = pr.read_gtf(gtf_file)
        n = len(self.intron_ranges)

        #  1. hgnc_symbol 
        genes_df = (
            gtf[gtf.Feature == "gene"]
            .df[["Chromosome", "Start", "End", "Strand", "gene_name"]]
            .drop_duplicates()
            .assign(Strand=lambda d: d["Strand"].astype(str))
        )
        genes_pr = pr.PyRanges(genes_df)

        junc_df = self.intron_ranges[["Chromosome", "Start", "End", "Strand"]].copy()
        junc_df["Start"] = junc_df["Start"] - 1  # converting 1-based (R/GRanges) to 0-based (pyranges)
        junc_df["junction_idx"] = np.arange(n)
        junctions_pr = pr.PyRanges(junc_df)
        strand_kw = {"strandedness": "same"} if (junctions_pr.stranded and genes_pr.stranded) else {}
        joined = junctions_pr.join(genes_pr, **strand_kw)
        if not joined.df.empty:
            symbol_map = (
                joined.df
                .groupby("junction_idx")["gene_name"]
                .apply(lambda x: ";".join(sorted(set(x.dropna()))))
            )
            self.intron_ranges[feature_name] = pd.Series(np.arange(n)).map(symbol_map).values
        else:
            self.intron_ranges[feature_name] = pd.NA

        #  2. annotatedJunction 
        # All unique reference introns from GTF
        known_introns_df = (
            gtf.features.introns(by="transcript")
            .df[["Chromosome", "Start", "End", "Strand", "gene_name"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(Strand=lambda d: d["Strand"].astype(str))
        )
        # FDS junctions that exactly match a reference intron
        exact_hits = junc_df.merge(
            known_introns_df, on=["Chromosome", "Start", "End", "Strand"], how="inner"
        )
        known_idx = exact_hits["junction_idx"].values

        # Median split-read count per junction across samples
        median_counts = np.median(self.K.values, axis=1)  # (n_junctions,)

        # Reference anno_pr: known FDS junctions sorted by median count descending
        is_known = np.zeros(n, dtype=bool)
        is_known[known_idx] = True
        anno_df = junc_df[is_known][["Chromosome", "Start", "End", "Strand"]].copy()
        # Add median counts where the GTF intron was observed in FDS
 
        anno_df["medianCount"] = median_counts[is_known]
        anno_df["fds_order"] = np.where(is_known)[0]
        
        annotations = np.full(n, "none", dtype=object)
        anno_pr = pr.PyRanges(anno_df)
        strand_kw["how"] = "left" #NEW

        ov_df = junctions_pr.join(anno_pr, **strand_kw).df  # right coords become Start_b / End_b
        # Keeping only the highest-count reference intron per junction

        # Then sort by medianCount first, fds_order second
        ov_df = ov_df.sort_values(["medianCount", "fds_order"],ascending=[False, True])   # highest count first, lowest FDS index first for ties
        print("Before choosing in ov_df: ")
        print(ov_df[(ov_df["Start"] == 35140132) &  (ov_df["End"] == 35144342) ])
        ov_df = ov_df.drop_duplicates(subset="junction_idx", keep="first")
        print("After choosing in ov_df: ")
        print(ov_df[(ov_df["Start"] == 35140132) &  (ov_df["End"] == 35144342) ])
        #ov_df = ov_df.sort_values("medianCount", ascending=False)
        #ov_df = ov_df.drop_duplicates(subset="junction_idx", keep="first")

        s = ov_df["Start"] == ov_df["Start_b"]
        e = ov_df["End"]   == ov_df["End_b"]
        ov_df["annotatedJunction"] = np.select(
            [s & e, s & ~e, ~s & e],
            ["both", "start", "end"],
            default="none",
        )
        idx = ov_df["junction_idx"].astype(int).values
        annotations[idx] = ov_df["annotatedJunction"].values

        self.intron_ranges["annotatedJunction"] = annotations
    
    def calculate_counts(self) -> dict[str, pd.DataFrame]:
        # returns samples* junctions
        nonsplitProportion = self.nonsplitCounts / self.N

        def broadcast(series: pd.Series) -> pd.DataFrame:
            """Repeat a per-junction scalar across all sample columns."""
            return pd.DataFrame({col: series for col in self.samples_cols}, index=self.N.index)

        return {
            'counts':                        self.K.T,
            'totalCounts':                   self.N.T,
            'nonsplitCounts':                self.nonsplitCounts.T,
            'nonsplitProportion':            nonsplitProportion.T,
            'meanCounts':                    broadcast(self.K.mean(axis=1)).T,
            'meanTotalCounts':               broadcast(self.N.mean(axis=1)).T,
            'nonsplitProportion_99quantile': broadcast(
                pd.Series(np.quantile(nonsplitProportion.values, 0.99, axis=1),
                          index=self.N.index)
            ).T,
        }

    def get_delta_psi(self, pseudocount=1):
        """
        Compute delta psi = jaccard index - median of the jaccard index across samples for each junction

        -------
        jaccard : DataFrame [samples x junctions]
        delta_psi : DataFrame [samples x junctions]
        """
        jaccard = (self.K + pseudocount) / (self.N + 2 * pseudocount)
        row_median = jaccard.median(axis=1) 
        delta_psi = jaccard.subtract(row_median, axis=0)
        return jaccard.T, delta_psi.T


