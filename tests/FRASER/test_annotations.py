# import pandas as pd

# def test_junction_annotations(fraser_dataset, GTF_PATH="/s/project/py_fraser/PROTRIDER/sample_data/gencode_annotation_trunc.gtf"):
#     n_before = len(fraser_dataset.intron_ranges)
#     fraser_dataset.annotate_junctions(GTF_PATH)
#     assert fraser_dataset.intron_ranges is not None
#     assert 'hgnc_symbol' in fraser_dataset.intron_ranges.columns, "intron_ranges should contain 'hgnc_symbol' column"
#     assert 'annotatedJunction' in fraser_dataset.intron_ranges.columns, "intron_ranges should contain 'annotatedJunction' column"
    
#     assert len(fraser_dataset.intron_ranges) == n_before, "Number of junctions should not change after annotation"

#     result_first = fraser_dataset.intron_ranges[["hgnc_symbol", "annotatedJunction"]].copy()
#     fraser_dataset.annotate_junctions(GTF_PATH)
#     result_second = fraser_dataset.intron_ranges[["hgnc_symbol", "annotatedJunction"]].copy()
#     pd.testing.assert_frame_equal(result_first, result_second)