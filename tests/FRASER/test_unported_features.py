"""
Placeholders for features with no Python counterpart yet.

Each skip below documents one feature that has no corresponding behavior
in protrider to test against. Kept as explicit skips (not simply omitted)
so "is X covered?" has one clear, greppable answer, and so
each turns into a real test automatically once the feature is ported.
"""
import pytest


@pytest.mark.skip(reason="Result.plot_pvals()/plot_aberrant_per_sample() are PROTRIDER/OUTRIDER-shaped "
                          "(hardcode 'proteinID', assume dataset.raw_data/dataset.data as intensities) "
                          "and are not adapted for FRASER junction-level output yet")
def test_plot_junction_dist():
    """Per-psiType splicing distribution plots."""


@pytest.mark.skip(reason="Result.plot_aberrant_per_sample()/plot_expected_vs_observed() are not "
                          "adapted for FRASER junction/gene-level output yet")
def test_plot_sample_results():
    """Per-sample aberrant splicing result plots."""


@pytest.mark.skip(reason="no seqlevels-style conversion (renaming chromosome/seqlevel naming "
                          "conventions, e.g. 'chr1' <-> '1') exists; the Python port assumes the "
                          "GTF and split/unsplit read files already agree on seqlevels style")
def test_update_seqlevels_style():
    """Chromosome/seqlevel naming-convention conversion."""


@pytest.mark.skip(reason="no equivalent of building UCSC/IGV/gnomAD hyperlink columns for a "
                          "results table exists in protrider")
def test_create_full_link_table():
    """External-browser (UCSC/IGV/gnomAD) hyperlink columns for a results table."""
