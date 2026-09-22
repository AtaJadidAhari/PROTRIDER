# PROTRIDER

PROTRIDER is an autoencoder-based method to call protein outliers from mass spectrometry-based proteomics datasets.

Have a look at our [paper](https://doi.org/10.1093/bioinformatics/btaf628) for information about our work.

## Installation

### Prerequisites

PROTRIDER was trained and tested using Python 3.14 on a Linux system. We recommend installing and running PROTRIDER on a dedicated conda environment. To create and activate the conda environment run the following commands:

```
conda create --name protrider_env python=3.14
conda activate protrider_env
```

More information on conda environments can be found in Conda's user guide.


To install PROTRIDER run the following command inside the root directory:

```
pip install .
```

To test the installation run 

```
protrider --help
```

## Usage

### Input files

- **Protein intensities**: CSV, TSV, or Parquet file
  - **File format**: Columns represent **samples**, rows represent **proteins** (wide format)
  - Example: `samples/data/protrider_sample_dataset.tsv`
- **Sample annotation** (optional): CSV or tab-separated file containing known covariates
  - Format: Each row represents a sample
  - Example: `samples/data/sample_annotations.tsv`

An example dataset can be found in this repository under `samples/data/`.

### OUTRIDER RNA-count mode

For `analysis: outrider`, supply integer RNA-seq counts and use `genes_as_rows`
(genes × samples) or `genes_as_columns` (samples × genes). Zeros are valid
counts. OUTRIDER uses `log((count + 1) / size_factor)` as input and the
negative-binomial expected count `size_factor * exp(decoder_output)` throughout
training and scoring. FPKM filtering uses `fpkm_percentile: 0.95` by default,
which retains genes above the cutoff in the upper 5% of samples.
When autoencoder training is disabled, the PCA-only path additionally fits the per-gene mean multiplier used by the original OUTRIDER PCA correction.

Set `outrider_precision` to `float32` (default) or `float64`. The selected precision is used for OUTRIDER autoencoder preprocessing, model parameters, dispersion fitting, inference, and statistical result arrays. The autoencoder uses the natural-log transform. OUTRIDER supports OHT or a fixed latent dimension, and currently rejects grid-search injection and cross-validation because those count-scale paths are not implemented consistently.

OHT follows [R OUTRIDER's `estimateBestQ`](https://github.com/gagneurlab/OUTRIDER/blob/b5dabe389fbcbead6de7d8c0287a194dcba8fe41/R/method-estimateBestQ.R): it normalizes counts by size factors, computes `log2((normalized_count + 1) / (gene_mean + 1))`, and standardizes each gene using its sample standard deviation before SVD. It includes the Marchenko–Pastur median correction in the threshold, permits `q = 1`, and falls back to `q = 2` only when no singular value exceeds the threshold. OHT uses double precision independently of training precision; its SVD is separate from the autoencoder's PCA initialization. As in R, genes with undefined standardized values (such as zero-variance genes) cause OHT to fail. A fixed `find_q_method` bypasses OHT.

Existing checkpoints retain their saved `q`. Use a fresh output directory without a checkpoint to rerun OHT selection. These OHT changes apply only to `analysis: outrider`; PROTRIDER and FRASER keep their existing selection methods.

Set `outrider_theta_fit_interval` to a positive integer to control theta fitting during training (default: `1`, every epoch). For example, `10` fits theta after epochs 10, 20, 30, etc. Between fits, training and loss evaluation use the current theta, starting with the robust moments estimate. Theta is always refitted after restoring the best model weights and again for final statistics, even if training ends before the next scheduled fit.

OUTRIDER early stopping is optional and disabled by default. Enable it with `outrider_early_stopping: true`; `outrider_early_stopping_min_epochs`, `outrider_early_stopping_patience`, and `outrider_early_stopping_min_delta` control when training stops. It monitors the full-cohort negative-binomial NLL every epoch using the current theta, restores the weights from the epoch with the lowest NLL, and refits theta for those restored weights. Patience is counted in epochs, so early stopping can occur between theta fits. `n_epochs` remains the maximum number of epochs.

OUTRIDER outputs `expected_counts.csv` and `residuals.csv`; the latter is
observed count minus expected count. `df_expected_counts` and
`df_raw_residuals` expose the same matrices in the Python API. OUTRIDER
adjusts p-values independently for each sample across its genes.
The final checkpoint is written after the definitive full-cohort dispersion fit and records theta, effective mean scale, size factors, identifiers, filtering metadata, and configuration.

### Configuration file

To run PROTRIDER, a configuration file needs to be provided. This can be adapted from the configuration file provided in this code repo (`config.yaml`). User options include

- `out_dir`: Path to the directory to store output files.
- `input_intensities`: Csv input file containing the protein intensities.
- `cov_used`: List of column names contained in the sample annotation file to be included as known covariates.
- `find_q_method`: Method to determine latent space dimension of autoencoder.
- `pval_dist`: Distribution (Gaussian or Student's t-test) for P-value calculation.

### Running PROTRIDER from the command line

Run PROTRIDER using the following command: 

```bash
protrider run --config <config_path>
```

All input and output paths (including `input_intensities`, `sample_annotation`, and `out_dir`) should be specified in the configuration file.

To generate plots with PROTRIDER, use the following command (specify one or more plot types as needed):
```bash
protrider plot --plot_type <plot_types> --config <config_path> --out_dir <out_dir>
```
#### Plot options

You can specify one or more plot types using the `--plot_type` option:

- `training_loss`: Plot training loss history
- `aberrant_per_sample`: Plot number of aberrant proteins per sample
- `pvals`: Plot the p-value plots
- `encoding_dim`: Plot the encoding dimension search plot
- `expected_vs_observed`: Plot expected vs observed protein intensity for a specific protein (requires `--protein_id`)
- `all`: Equivalent to specifying all of the above except `expected_vs_observed`.

Example:
```
protrider plot --plot_type pvals --config <config_path>
```

To plot expected vs observed for a specific protein:
```
protrider plot --plot_type expected_vs_observed --protein_id <protein_id> --config <config_path>
```


#### Output files

PROTRIDER generates several output files in the specified output directory:

- `pvals.csv`: P-values for each protein in each sample
- `pvals_adj.csv`: Adjusted p-values (FDR correction)
- `zscores.csv`: Z-scores for each protein in each sample
- `residuals.csv`: Model residuals
- `log2fc.csv`: Log2 fold changes
- `fc.csv`: Fold changes
- `output.csv`: Combined long-format output (when using `format='long'`)
- `config.yaml`: Saved configuration for reproducibility
- `model_info.yaml`: Model metadata (latent dimension, learning rate, etc.)


### Using PROTRIDER as a Python package

PROTRIDER can also be used directly as a Python package for more flexibility:

```python
import protrider
import pandas as pd

# File format: columns = samples, rows = proteins
config = protrider.ProtriderConfig(
    out_dir='output/',
    input_intensities='data/protein_intensities.csv',  # Columns = samples
    sample_annotation='data/sample_annotations.csv',
    index_col='protein_ID',
    cov_used=['AGE', 'SEX'],
    n_epochs=100
)

# Run PROTRIDER
result, model_info = protrider.run(config)

# Save results in different formats
result.save(config.out_dir, format='wide')  # Individual CSV files (pvals.csv, zscores.csv, etc.)
result.save(config.out_dir, format='long')  # Single combined CSV (output.csv)

# Save model info and config
model_info.save(config.out_dir)  # Saves additional_info.csv, train_losses.csv
config.save(config.out_dir)      # Saves config.yaml

# Generate plots (out_dir is optional - omit to get plot objects without saving)
model_info.plot_training_loss(config.out_dir)
result.plot_aberrant_per_sample(config.out_dir)
hist_plot, qq_plot = result.plot_pvals(config.out_dir)
result.plot_expected_vs_observed('protein_123', config.out_dir)

# Access results as DataFrames for further analysis
pvals = result.df_pvals           # P-values
pvals_adj = result.df_pvals_adj   # Adjusted p-values
zscores = result.df_Z             # Z-scores
residuals = result.df_res         # Residuals
log2fc = result.log2fc            # Log2 fold changes
fc = result.fc                    # Fold changes
```

## Citation
If you use this tool, please cite the original paper:
```
@article{10.1093/bioinformatics/btaf628,
    author = {Klaproth-Andrade, Daniela and Scheller, Ines F and Tsitsiridis, Georgios and Loipfinger, Stefan and Mertes, Christian and Smirnov, Dmitrii and Prokisch, Holger and Yépez, Vicente A and Gagneur, Julien},
    title = {PROTRIDER: Protein abundance outlier detection from mass spectrometry-based proteomics data with a conditional autoencoder},
    journal = {Bioinformatics},
    pages = {btaf628},
    year = {2025},
    month = {11},
    abstract = {Detection of gene regulatory aberrations enhances our ability to interpret the impact of inherited and acquired genetic variation for rare disease diagnostics and tumor characterization. While numerous methods for calling RNA expression outliers from RNA-sequencing data have been proposed, the establishment of protein expression outliers from mass spectrometry data is lacking.Here, we propose and assess various modeling approaches to call protein expression outliers across three datasets from rare disease diagnostics and oncology. We use as independent evidence the enrichment for outlier calls in matched RNA-seq samples and the enrichment for rare variants likely disrupting protein expression. We show that controlling for hidden confounders and technical covariates, while simultaneously modeling the occurrence of missing values, is largely beneficial and can be achieved using conditional autoencoders. Moreover, we find that the differences between experimental and fitted log-transformed intensities by such models exhibit heavy tails that are poorly captured with the Gaussian distribution and report stronger statistical calibration when instead using the Student’s t-distribution. Our resulting method, PROTRIDER, outperformed baseline approaches based on raw log-intensities Z-scores, PCA, and isolation-based anomaly detection with Isolation forests. The application of PROTRIDER reveals significant enrichments of AlphaMissense pathogenic variants in protein expression outliers. Overall, PROTRIDER provides a method to confidently identify aberrantly expressed proteins applicable to rare disease diagnostics and cancer proteomics.PROTRIDER is freely available at github.com/gagneurlab/PROTRIDER and also available on Zenodo under the DOI zenodo.15569781.Supplementary data are available at Bioinformatics online.},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf628},
    url = {https://doi.org/10.1093/bioinformatics/btaf628},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaf628/65416092/btaf628.pdf},
}
```
