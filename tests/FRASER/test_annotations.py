def test_annotation_matches_r_reference(fraser_dataset_annotated, r_annotations):
    ds = fraser_dataset_annotated
    passed_ref = r_annotations[r_annotations["passed"]]

    # Match by (startID, endID) since both datasets were filtered from the same underlying junctions.
    ref_by_pos = passed_ref.set_index(["startID", "endID"])
    ours = ds.intron_ranges.set_index(["StartId", "EndId"])

    common_idx = ours.index.intersection(ref_by_pos.index)
    assert len(common_idx) == len(ours), "Every filtered junction should be matchable to the reference by position"

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
