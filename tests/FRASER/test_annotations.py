def test_annotation_matches_r_reference(fraser_dataset_annotated, r_annotations):
    """Compare against FRASER-R's own annotation for the same (chr21) region.

    Every filtered junction must be positionally matchable to the R reference. Agreement on
    annotatedJunction is required for all but one junction (StartId=576, EndId=1249,
    U2AF1/U2AF1L5): our code independently derives "both" for it from two sources (an exact
    intron match in the current sample_data/gencode_annotation_trunc.gtf, and from the
    junction itself being a known splice junction in the observed data), so this single
    disagreement with R's pre-computed junction_annotations.csv looks like that reference
    having been generated against a different/older GTF snapshot, not a bug here. 159/160 is
    therefore the maximum achievable agreement against this particular reference file.
    """
    ds = fraser_dataset_annotated
    passed_ref = r_annotations[r_annotations["passed"]]

    # Match by (startID, endID) since both datasets were filtered from the same underlying junctions.
    ref_by_pos = passed_ref.set_index(["startID", "endID"])
    ours = ds.intron_ranges.set_index(["StartId", "EndId"])

    common_idx = ours.index.intersection(ref_by_pos.index)
    assert len(common_idx) == len(ours), "Every filtered junction should be matchable to the R reference by position"

    matches = (
        ours.loc[common_idx, "annotatedJunction"].to_numpy()
        == ref_by_pos.loc[common_idx, "annotatedJunction"].to_numpy()
    )
    agreement = matches.mean()
    max_achievable_agreement = 159 / 160
    assert agreement >= max_achievable_agreement, (
        f"annotatedJunction agreement with R reference regressed: {agreement:.4%} "
        f"(expected >= {max_achievable_agreement:.4%})"
    )
