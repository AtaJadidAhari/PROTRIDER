"""Export junction (split reads) and splice site (unsplit reads) counts from a
FraserDataSet (fds-object.RDS + HDF5 assay files) to CSV, matching the schema of
the R reference export (as.data.table(rowRanges(fds)) / rowData(nonSplicedReads(fds))).
"""

import argparse
import glob
import os
import warnings

import h5py
import numpy as np
import pandas as pd
from rds2py import parse_rds
from rds2py.generics import _dispatcher
from rds2py.rdsutils import get_class


def _read_factor(robject, **kwargs):
    """R factor -> list of labels. Coerces scalar `data`/`lengths` to length-1
    sequences: rds2py's `read_factor` iterates them directly and raises TypeError
    on some run-length-encoded factors whose `data` comes back as a bare scalar.
    """
    data = robject["data"]
    data = [data] if np.isscalar(data) else data

    levels = _dispatcher(robject["attributes"]["levels"], **kwargs) \
        if "levels" in robject["attributes"] else None
    level_vec = [levels[int(x) - 1] if pd.notna(x) else None for x in data]

    if "lengths" in robject["attributes"]:
        lengths = _dispatcher(robject["attributes"]["lengths"], **kwargs)
        lengths = [lengths] if np.isscalar(lengths) else lengths
    else:
        lengths = [1] * len(data)

    return [v for i, n in enumerate(lengths) for v in [level_vec[i]] * int(n)]


def _read_mcols(robject, n_features=None, **kwargs):
    """DFrame (elementMetadata) -> dict of columns, skipping any column that fails
    to parse or comes back the wrong length (e.g. via `_read_factor`'s bug, or
    rds2py's dispatcher swallowing a 0-column DFrame's TypeError and returning an
    unusable raw dict instead of raising).
    """
    if robject is None or get_class(robject) != "DFrame":
        return {}

    list_data = robject["attributes"]["listData"]
    names_robj = list_data["attributes"].get("names")
    col_names = _dispatcher(names_robj, **kwargs) if names_robj is not None else None
    if not col_names:
        return {}

    result = {}
    for idx, colname in enumerate(col_names):
        col_robj = list_data["data"][idx]
        try:
            values = _read_factor(col_robj, **kwargs) if get_class(col_robj) == "factor" \
                else _dispatcher(col_robj, **kwargs)
            values = np.asarray(values)
            if n_features is not None and values.shape[0] != n_features:
                raise ValueError(f"parsed length {values.shape[0]} != expected {n_features}")
        except Exception as e:
            warnings.warn(f"Skipping mcols column '{colname}': {e}", RuntimeWarning)
            continue
        result[colname] = values
    return result


def _granges_to_pandas(robject, **kwargs):
    """R GenomicRanges -> pandas DataFrame (seqnames/start/end/width/strand + mcols),
    without building a genomicranges.GenomicRanges/BiocFrame object, since that
    construction raises when mcols fails to parse as a BiocFrame (see `_read_mcols`).
    """
    cls = get_class(robject)
    if cls not in ("GenomicRanges", "GRanges"):
        raise TypeError(f"obj is not 'GenomicRanges', but is `{cls}`.")

    ranges_attrs = robject["attributes"]["ranges"]["attributes"]
    start = np.asarray(_dispatcher(ranges_attrs["start"], **kwargs))
    width = np.asarray(_dispatcher(ranges_attrs["width"], **kwargs))

    df = pd.DataFrame({
        "seqnames": np.asarray(_dispatcher(robject["attributes"]["seqnames"], **kwargs)),
        "start": start,
        "end": start + width - 1,
        "width": width,
        "strand": np.asarray(_dispatcher(robject["attributes"]["strand"], **kwargs)),
    })

    mcols = _read_mcols(robject["attributes"].get("elementMetadata"), n_features=len(df), **kwargs)
    for colname, values in mcols.items():
        df[colname] = values
    return df


def _read_hdf5_assay(fds_dir, assay_name, n_features=None):
    """Read a FRASER HDF5-backed assay matrix.

    Corrects orientation against `n_features` (R/rhdf5 and h5py don't always agree
    on (features x samples) vs transposed), and uses the real sample names from R's
    dimnames group when available (".<assay>_dimnames" key "2") instead of
    synthetic labels, so columns match a reference R export's real sample IDs.
    """
    candidates = glob.glob(os.path.join(fds_dir, f"{assay_name}*.h5")) \
        or glob.glob(os.path.join(fds_dir, "*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No .h5 file found for assay '{assay_name}' in {fds_dir}")

    with h5py.File(candidates[0], "r") as f:
        dset_name = assay_name if assay_name in f else list(f.keys())[0]
        data = f[dset_name][:]

        sample_names = None
        dimnames_key = f".{assay_name}_dimnames"
        if dimnames_key in f and hasattr(f[dimnames_key], "keys") and "2" in f[dimnames_key]:
            sample_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                             for x in f[dimnames_key]["2"][:]]

    if data.ndim == 2 and n_features is not None:
        if data.shape[1] == n_features and data.shape[0] != n_features:
            data = data.T
        elif n_features not in data.shape:
            raise ValueError(
                f"'{assay_name}' has shape {data.shape}, but neither dimension "
                f"matches the expected number of features ({n_features})."
            )

    if data.ndim != 2:
        return pd.DataFrame({assay_name: data})

    columns = sample_names if sample_names and len(sample_names) == data.shape[1] \
        else [f"{assay_name}_sample{i}" for i in range(data.shape[1])]
    return pd.DataFrame(data, columns=columns)


def convert_fraser_dataset(fds_dir, output_path, dataset_name, output_s=None, output_u=None):
    output_s = output_s or os.path.join(output_path, f"{dataset_name}_s.csv")
    output_u = output_u or os.path.join(output_path, f"{dataset_name}_u.csv")

    attrs = parse_rds(os.path.join(fds_dir, "fds-object.RDS"))["attributes"]

    row_ranges_df = _granges_to_pandas(attrs["rowRanges"])
    splice_sites_df = _granges_to_pandas(attrs["nonSplicedReads"]["attributes"]["rowRanges"]) \
        .drop(columns=["seqnames", "start", "end", "width", "strand"], errors="ignore")

    junction_counts = _read_hdf5_assay(fds_dir, "rawCountsJ", n_features=len(row_ranges_df))
    splice_site_counts = _read_hdf5_assay(fds_dir, "rawCountsSS", n_features=len(splice_sites_df))

    # concat positionally (reset_index): these frames share no meaningful index, and
    # aligning mismatched/duplicate default RangeIndexes on axis=1 can silently explode
    # into an outer-join cross-product.
    junction_df = pd.concat(
        [junction_counts.reset_index(drop=True),
         row_ranges_df[["seqnames", "start", "end", "width", "strand",
                        "startID", "endID", "intron_motif"]].reset_index(drop=True)],
        axis=1,
    )
    splice_site_df = pd.concat(
        [splice_site_counts.reset_index(drop=True), splice_sites_df.reset_index(drop=True)],
        axis=1,
    )

    splice_site_df.to_csv(output_u, index=False)
    junction_df.to_csv(output_s, index=False)
    print(f"Done. Files written to: {output_u} and {output_s}")


def main():
    parser = argparse.ArgumentParser(description="Export FRASER counts to CSV")
    parser.add_argument("-i", "--fds_dir", required=True, help="Path to FRASER dataset directory (containing fds-object.RDS)")
    parser.add_argument("-o", "--output_path", required=True, help="Path to output directory")
    parser.add_argument("-n", "--dataset_name", required=True, help="Name of the FRASER dataset")
    parser.add_argument("--output_s", default=None, help="Path to junction (split reads) output file")
    parser.add_argument("--output_u", default=None, help="Path to splice site (unsplit reads) output file")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    convert_fraser_dataset(args.fds_dir, args.output_path, args.dataset_name, args.output_s, args.output_u)


if __name__ == "__main__":
    main()
