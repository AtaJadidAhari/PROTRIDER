import numpy as np
import pandas as pd

def test_split_and_unsplit_read_counts(fraser_dataset):
    assert isinstance(fraser_dataset.split_reads, pd.DataFrame), "split_reads should be a DataFrame"
    assert isinstance(fraser_dataset.unsplit_reads, pd.DataFrame), "unsplit_reads should be a DataFrame"

def test_nominator_and_denominator_of_jaccard_index(fraser_dataset):
    n_junctions = len(fraser_dataset.split_reads)
    n_samples   = len(fraser_dataset.samples_cols)

    # type
    assert isinstance(fraser_dataset.K, pd.DataFrame), "K should be a DataFrame"
    assert isinstance(fraser_dataset.N, pd.DataFrame), "N should be a DataFrame"
    
    # shape
    assert fraser_dataset.K.shape == (n_junctions, n_samples), "K should have shape (n_junctions, n_samples)"
    assert fraser_dataset.N.shape == (n_junctions, n_samples), "N should have shape (n_junctions, n_samples)"

    # columns
    assert list(fraser_dataset.K.columns) == list(fraser_dataset.samples_cols), "K columns should match sample columns"
    assert list(fraser_dataset.N.columns) == list(fraser_dataset.samples_cols), "N columns should match sample columns"

    # values
    assert (fraser_dataset.K.values >= 0).all(), "K values should be non-negative"
    assert (fraser_dataset.N.values >= 0).all(), "N values should be non-negative"

    # K should be less than or equal to N
    assert (fraser_dataset.N.values >= fraser_dataset.K.values).all(), "N values should be greater than or equal to K values"

def test_jaccard_index(fraser_dataset):
    n_samples   = len(fraser_dataset.samples_cols)
    n_filtered  = int(fraser_dataset.passed_expression.sum())
    # type
    assert isinstance(fraser_dataset.jaccard_index, pd.DataFrame), "jaccard_index should be a DataFrame"

    # shape
    assert fraser_dataset.jaccard_index.shape == (n_filtered, n_samples), "jaccard_index should have shape (n_filtered, n_samples)"
    
    # values: TODO -> What if N zero
    assert (fraser_dataset.jaccard_index >= 0).all() and (fraser_dataset.jaccard_index <= 1).all(), "jaccard_index values should be between 0 and 1"

def test_passed_expression_and_data(fraser_dataset):
    n_junctions = len(fraser_dataset.split_reads)
    # type
    assert isinstance(fraser_dataset.passed_expression, np.ndarray), "passed_expression should be a numpy array"
    assert fraser_dataset.passed_expression.dtype == bool, "passed_expression should be a boolean array"

    # shape
    assert len(fraser_dataset.passed_expression) == n_junctions, "passed_expression should have the same length as split_reads"
    # at least some junctions pass
    assert fraser_dataset.passed_expression.any(), "At least some junctions should pass expression filtering"

def test_centered_log_data_noNA(fraser_dataset):
    n_samples   = len(fraser_dataset.samples_cols)
    n_junctions = fraser_dataset.split_reads.shape[0]

    # type
    assert isinstance(fraser_dataset.centered_log_data_noNA, np.ndarray), "centered_log_data_noNA should be a numpy array"
    
    # shape
    # TODO check : (n_samples, n_filtered)
    assert fraser_dataset.centered_log_data_noNA.shape == (n_samples, n_junctions), "centered_log_data_noNA should have shape (n_samples, n_junctions)"
