"""
Placeholder tests for BAM-level read counting.

FraserDataset does not implement BAM-level read counting: it starts from
already-counted split_reads/unsplit_reads TSVs (see sample_data/fraser/split_reads.tsv,
unsplit_reads.tsv), which is the K/N-computation stage tested in test_data.py.

These are kept as explicit skips - rather than simply omitting the file - so
that "is BAM counting covered?" has one clear, greppable answer, and so the
tests turn into real ones automatically if/when BAM counting is ported.
"""
import pytest


@pytest.mark.skip(reason="BAM-level split-read counting is not implemented; "
                          "FraserDataset starts from precomputed split/unsplit read TSVs")
def test_count_junctions_from_bam():
    """Paired-end vs single-end split-read counting from a BAM file, compared against
    manually counted positions."""


@pytest.mark.skip(reason="BAM-level strand-specific counting is not implemented")
def test_strand_specific_counting_from_bam():
    """Stranded split/non-split read counting from a BAM file."""


@pytest.mark.skip(reason="BAM-level non-split (splice-site anchor) counting is not implemented")
def test_min_anchor_length_for_nonsplit_counting():
    """Non-split read counting with a minimum splice-site anchor length."""


@pytest.mark.skip(reason="BAM-level counting is not implemented; only the Jaccard metric is supported")
def test_psi_values_from_bam_counts():
    """psi3/psi5/theta splicing metrics derived directly from BAM-counted reads. The Python
    port only implements the Intron Jaccard Index metric (FraserDataset.jaccard_index)."""
