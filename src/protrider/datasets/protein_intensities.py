from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import copy
from pydeseq2.preprocessing import deseq2_norm
import logging

logger = logging.getLogger(__name__)


def _numeric_overrides(path, sep, compression, nrows=20):
    """Pin numeric columns' dtypes from a preview: whole-number columns become
    int32 (instead of defaulting to float64), other numeric columns float32."""
    preview = pd.read_csv(path, sep=sep, compression=compression, nrows=nrows)
    return {
        col: "int32" if preview[col].dropna().mod(1).eq(0).all() else "float32"
        for col in preview.select_dtypes("number").columns
    }


def read(input_intensities: str, index_col: str = None, input_format: str = "proteins_as_rows") -> pd.DataFrame:
    """Read protein intensities from a file.
    
    Args:
        input_intensities: Path to file (CSV, TSV, or Parquet)
        index_col: Name of the index column containing protein IDs
        input_format: Format of the input file:
                     - "proteins_as_rows": proteins are rows, samples are columns (default)
                     - "proteins_as_columns": samples are rows, proteins are columns
    
    Returns:
        pd.DataFrame: Protein intensities with samples as rows and proteins as columns
    """
    intensities = []
    for input_intensity in input_intensities:
        suffixes = Path(input_intensity).suffixes
        compression = None
        if suffixes[-1] == '.gz':
            compression = 'gzip'
            suffixes = suffixes[:-1]
        if suffixes[-1] == '.csv':
            sep = ','
            temp_data = pd.read_csv(input_intensity, sep=sep, compression=compression,
                                     dtype=_numeric_overrides(input_intensity, sep, compression))
            if index_col is not None:
                temp_data = temp_data.set_index(index_col)
        elif suffixes[-1] == '.tsv':
            sep = '\t'
            temp_data = pd.read_csv(input_intensity, sep=sep, compression=compression,
                                     dtype=_numeric_overrides(input_intensity, sep, compression))
            if temp_data.shape[1] == 1:
                sep = ','
                temp_data = pd.read_csv(input_intensity, sep=sep, compression=compression,
                                         dtype=_numeric_overrides(input_intensity, sep, compression))
            if index_col is not None:
                temp_data = temp_data.set_index(index_col)
        elif suffixes[-1] == '.parquet':
            temp_data = pd.read_parquet(input_intensity).set_index(index_col)
        else:
            raise ValueError(f"Unsupported file type: {suffixes[-1]}")
        intensities.append(temp_data)
    data = pd.concat(intensities, axis=1)

    omics = '(samples , proteins)' if input_format in ['proteins_as_rows', 'proteins_as_columns'] else ''
    # Transpose if needed to get samples as rows, proteins as columns
    if input_format == "proteins_as_rows":
        data = data.T
        data.index.names = ['sampleID']
        data.columns.name = 'proteinID'
    elif input_format == "proteins_as_columns":
        # Already in the correct format, just set names
        data.index.name = 'sampleID'
        data.columns.name = 'proteinID'
    elif input_format == 'introns_as_rows': # TODO 
        pass
    else:
        raise ValueError(f"Invalid input_format: {input_format}. Must be 'proteins_as_rows' or 'proteins_as_columns'")
    
    logger.info(f'Finished reading raw data with shape: {data.shape} {omics}')

    return data


def preprocess_protein_intensities(data, log_func, maxNA_filter):
    """Preprocess protein intensities data.

    Args:
        data (pd.DataFrame): Input protein intensities data.
        log_func (callable): Function to apply log transformation.
        maxNA_filter (float): Maximum allowed proportion of NA values.

    Returns:
        tuple: Processed protein intensities, filtered (no NAs) raw data, and size factors.
    """

    raw_data = None
    size_factors = None
    processed_data = data
    if log_func is not None:
        # replace 0 with NaN (for proteomics intensities)
        processed_data.replace(0, np.nan, inplace=True)

        # filter out proteins with too many NaNs
        filtered = np.mean(np.isnan(processed_data), axis=0)
        filtered_data = (processed_data.T[filtered <= maxNA_filter]).T
        logger.info(
            f"Filtering out {np.sum(filtered > maxNA_filter)} proteins with too many missing values. New shape: {data.shape}")
        raw_data = copy.deepcopy(processed_data)  ## for storing output

        # normalize data with deseq2
        size_factors = None
        deseq_out, size_factors = deseq2_norm(filtered_data.replace(np.nan, 0,
                                                                inplace=False))
        ### check that deseq2 worked, otherwise ignore
        if deseq_out.isna().sum().sum() == 0:
            processed_data = deseq_out
            processed_data.replace(0, np.nan, inplace=True)
            size_factors = size_factors

        # log data
        processed_data = log_func(processed_data)
    else:
        # filter out proteins with too many NaNs
        filtered = np.mean(np.isnan(processed_data), axis=0)
        processed_data = (processed_data.T[filtered <= maxNA_filter]).T
        logger.info(
            f"Filtering out {np.sum(filtered > maxNA_filter)} proteins with too many missing values. New shape: {processed_data.shape}")
        raw_data = copy.deepcopy(processed_data)  ## for storing output

    return processed_data, raw_data, size_factors



def merge_split_reads(split_reads: list, unsplit_reads_1: pd.DataFrame = None):
    """
    Returns (result_df, id_mapping) where id_mapping maps every old
    startID/endID value seen in the unmatched part of split_reads_2 to
    its final value in the merged result (identity if unchanged).
    """
    split_reads_1 = split_reads[0]
    split_reads_2 = split_reads[1]

    meta_cols = [c for c in split_reads_1.columns if c in split_reads_2.columns]
    sample_cols_2 = [c for c in split_reads_2.columns if c not in meta_cols]

    join_on = ["seqnames", "start", "end", "strand"]
    id_cols = ["startID", "endID"]

    # find which split_reads_2 rows match split_reads_1 on join_on
    key_check = split_reads_2[join_on].merge(split_reads_1[join_on].drop_duplicates(), on=join_on, how="left", indicator=True)
    is_matched = (key_check["_merge"] == "both").values

    sr2_matched = split_reads_2.loc[is_matched].copy()
    sr2_unmatched = split_reads_2.loc[~is_matched].copy()

    # Step 1: left-merge matched part onto split_reads_1
    sr2_sub = sr2_matched[join_on + sample_cols_2]
    merged = split_reads_1.merge(sr2_sub, on=join_on, how="left")
    merged[sample_cols_2] = merged[sample_cols_2].fillna(0).astype(int)

    id_mapping = {}

    # Step 2: build new rows for unmatched split_reads_2 rows
    if not sr2_unmatched.empty:
        new_rows = pd.DataFrame(0, index=sr2_unmatched.index, columns=merged.columns)
        for col in meta_cols:
            new_rows[col] = sr2_unmatched[col].values
        new_rows[sample_cols_2] = sr2_unmatched[sample_cols_2].values
        # sample_cols_1 stay 0 by construction

        # shared ID space: split_reads_1's startID/endID + unsplit_reads_1's spliceSiteID
        existing_ids = set(split_reads_1[id_cols[0]]) | set(split_reads_1[id_cols[1]])
        candidate_maxes = [
            split_reads_1[id_cols[0]].max(skipna=True),
            split_reads_1[id_cols[1]].max(skipna=True),
        ]
        if unsplit_reads_1 is not None:
            existing_ids |= set(unsplit_reads_1["spliceSiteID"])
            candidate_maxes.append(unsplit_reads_1["spliceSiteID"].max(skipna=True))
        max_id = max(candidate_maxes)

        for col in id_cols:
            new_vals = []
            for val in new_rows[col]:
                if val in id_mapping:
                    new_val = id_mapping[val]  # already resolved (e.g. seen as startID and endID)
                elif val in existing_ids:
                    max_id += 1
                    new_val = max_id
                    id_mapping[val] = new_val
                    existing_ids.add(new_val)
                else:
                    new_val = val  # unchanged, but still recorded
                    id_mapping[val] = new_val
                    existing_ids.add(new_val)
                new_vals.append(new_val)
            new_rows[col] = new_vals

        result = pd.concat([merged, new_rows], ignore_index=True, sort=False)
    else:
        result = merged

    return result, id_mapping


def merge_unsplit_reads(unsplit_reads: list, id_mapping: dict) -> pd.DataFrame:
    """
    Merges two unsplit_reads dataframes on spliceSiteID.
    - Matching spliceSiteID -> join sample_cols_2 onto unsplit_reads_1.
    - New spliceSiteID -> appended as a new row, with spliceSiteID replaced
      by its corresponding value in id_mapping (from merge_split_reads).
      Raises if no mapping entry exists for that ID.
    """
    unsplit_reads_1 = unsplit_reads[0]
    unsplit_reads_2 = unsplit_reads[1]

    meta_cols = [c for c in unsplit_reads_1.columns if c in unsplit_reads_2.columns]
    sample_cols_2 = [c for c in unsplit_reads_2.columns if c not in meta_cols]

    join_on = ["spliceSiteID"]

    # find which unsplit_reads_2 rows match unsplit_reads_1 on spliceSiteID
    key_check = unsplit_reads_2[join_on].merge(unsplit_reads_1[join_on].drop_duplicates(), on=join_on, how="left", indicator=True)
    is_matched = (key_check["_merge"] == "both").values

    ur2_matched = unsplit_reads_2.loc[is_matched].copy()
    ur2_unmatched = unsplit_reads_2.loc[~is_matched].copy()

    # Step 1: left-merge matched part onto unsplit_reads_1
    ur2_sub = ur2_matched[join_on + sample_cols_2]
    merged = unsplit_reads_1.merge(ur2_sub, on=join_on, how="left")
    merged[sample_cols_2] = merged[sample_cols_2].fillna(0).astype(int)

    # Step 2: build new rows for unmatched unsplit_reads_2 rows
    if not ur2_unmatched.empty:
        old_ids = ur2_unmatched["spliceSiteID"]
        missing = old_ids[~old_ids.isin(id_mapping.keys())]
        if not missing.empty:
            raise KeyError(
                f"No id_mapping entry found for spliceSiteID(s): {sorted(missing.unique().tolist())}. "
                "Every new spliceSiteID must correspond to a startID/endID handled in merge_split_reads."
            )

        new_rows = pd.DataFrame(0, index=ur2_unmatched.index, columns=merged.columns)
        for col in meta_cols:
            new_rows[col] = ur2_unmatched[col].values
        new_rows[sample_cols_2] = ur2_unmatched[sample_cols_2].values
        new_rows["spliceSiteID"] = old_ids.map(id_mapping).values
        # sample_cols_1 stay 0 by construction

        result = pd.concat([merged, new_rows], ignore_index=True, sort=False)
    else:
        result = merged

    return result