#!/usr/bin/env python3
"""Export junction (split reads) and splice site (unsplit reads) counts
from a FraserDataSet to CSV. Python port of fraser_export_script.R.

Loads the FraserDataSet via R's FRASER package (through rpy2) and writes
the same two CSVs as the original R script.
"""

import argparse
import os

from rpy2.robjects import default_converter, r, pandas2ri
from rpy2.robjects.packages import importr


def parse_args():
    parser = argparse.ArgumentParser(description="Export FRASER counts to CSV")
    parser.add_argument("-i", "--input_path", required=True, help="Path to FRASER dataset directory")
    parser.add_argument("-o", "--output_path", required=True, help="Path to output directory")
    parser.add_argument("-n", "--dataset_name", required=True, help="Name of the FRASER dataset")
    parser.add_argument("--output_s", default=None, help="Path to junction (split reads) output file")
    parser.add_argument("--output_u", default=None, help="Path to splice site (unsplit reads) output file")
    return parser.parse_args()


def main():
    args = parse_args()

    output_s_path = args.output_s or os.path.join(args.output_path, f"{args.dataset_name}_s.csv")
    output_u_path = args.output_u or os.path.join(args.output_path, f"{args.dataset_name}_u.csv")

    importr("FRASER")
    importr("data.table")

    r.assign("input_path", args.input_path)
    r.assign("dataset_name", args.dataset_name)

    fds = r(
        """
        fds <- loadFraserDataSet(dir=input_path, name=dataset_name)

        row_ranges      <- as.data.table(rowRanges(fds))
        junction_counts <- as.data.table(assays(fds)$rawCountsJ)
        junction_counts <- cbind(
          junction_counts,
          row_ranges[, c("seqnames", "start", "end", "width", "strand",
                         "startID", "endID", "intron_motif")]
        )

        splice_sites       <- rowData(nonSplicedReads(fds))
        splice_site_counts <- as.data.table(assays(fds)$rawCountsSS)
        splice_site_counts <- cbind(splice_site_counts, as.data.table(splice_sites))

        list(junction_counts = as.data.frame(junction_counts),
             splice_site_counts = as.data.frame(splice_site_counts))
        """
    )

    with (default_converter + pandas2ri.converter).context():
        junction_counts = fds.rx2("junction_counts")
        splice_site_counts = fds.rx2("splice_site_counts")

    splice_site_counts.to_csv(output_u_path, index=False)
    junction_counts.to_csv(output_s_path, index=False)

    print(f"Done. Files written to: {output_u_path} and {output_s_path}")


if __name__ == "__main__":
    main()
