from __future__ import annotations

import pandas as pd

from phase_evonet.r3.gnome_provenance import apply_human_attestation, classify_records, deterministic_review_sample


def _row(**overrides):
    row = {
        "transition_id": "00" * 16,
        "competitor_contextual_id": "ctx-0",
        "candidate_lineage_id": "lin-0",
        "identity_side": "target",
        "identity_snapshot": "2025-09-25",
        "material_id": "mp-1",
        "task_id": "mp-2",
        "entry_id": "mp-2",
        "thermo_type": "r2SCAN",
        "identity_workflow": "R2SCAN",
        "raw_builder_batch_id": None,
        "material_builder_license": None,
        "raw_builder_license": None,
        "provenance_history_names": "",
        "provenance_database_id_names": "",
        "provenance_keyword_hit": False,
        "r3_3r_roundtrip": True,
        "p3_2_roundtrip": True,
        "raw_material_roundtrip": True,
        "source_manifest_match": True,
    }
    row.update(overrides)
    return row


def test_direct_structured_gnome_requires_batch_license_and_roundtrips():
    frame = pd.DataFrame([_row(raw_builder_batch_id="gnome_r2scan_statics", material_builder_license="BY-NC", raw_builder_license="BY-NC")])
    result = classify_records(frame, {"gnome_r2scan_statics"})
    assert result.loc[0, "provenance_class"] == "A"
    assert bool(result.loc[0, "eligible_after_human_review"])


def test_keyword_or_date_alone_cannot_create_a():
    frame = pd.DataFrame([_row(provenance_keyword_hit=True, raw_builder_build_date="2025-04-03T00:00:00Z")])
    result = classify_records(frame, {"gnome_r2scan_statics"})
    assert result.loc[0, "provenance_class"] == "U"


def test_official_batch_with_by_c_conflict_is_unresolved():
    frame = pd.DataFrame([_row(raw_builder_batch_id="gnome_r2scan_statics", material_builder_license="BY-C", raw_builder_license="BY-C")])
    result = classify_records(frame, {"gnome_r2scan_statics"})
    assert result.loc[0, "provenance_class"] == "U"
    assert "conflicts" in result.loc[0, "classification_reason"]


def test_structured_other_batch_can_be_c():
    frame = pd.DataFrame([_row(raw_builder_batch_id="zbare_nrel_perovskites", material_builder_license="BY-C", raw_builder_license="BY-C")])
    result = classify_records(frame, {"gnome_r2scan_statics"})
    assert result.loc[0, "provenance_class"] == "C"


def test_exact_material_task_inheritance_is_b_not_a():
    frame = pd.DataFrame([
        _row(transition_id="01" * 16, competitor_contextual_id="ctx-a", raw_builder_batch_id="gnome_r2scan_statics", material_builder_license="BY-NC", raw_builder_license="BY-NC"),
        _row(transition_id="02" * 16, competitor_contextual_id="ctx-b", identity_snapshot="2024-12-18", raw_builder_batch_id=None, material_builder_license="BY-NC", raw_builder_license=None),
    ])
    result = classify_records(frame, {"gnome_r2scan_statics"})
    assert result.provenance_class.tolist() == ["A", "B"]


def test_missing_roundtrip_prevents_a_b_and_c():
    frame = pd.DataFrame([_row(raw_builder_batch_id="gnome_r2scan_statics", material_builder_license="BY-NC", raw_builder_license="BY-NC", p3_2_roundtrip=False)])
    result = classify_records(frame, {"gnome_r2scan_statics"})
    assert result.loc[0, "provenance_class"] == "U"


def test_review_sample_is_deterministic_and_blank_for_humans():
    rows = []
    for index in range(500):
        rows.append(_row(
            transition_id=f"{index:032x}", competitor_contextual_id=f"ctx-{index}",
            material_id=f"mp-{index}", task_id=f"task-{index}",
            identity_snapshot=("2024-12-18" if index % 2 else "2025-09-25"),
            raw_builder_batch_id=("gnome_r2scan_statics" if index % 3 == 0 else None),
            material_builder_license=("BY-NC" if index % 3 == 0 else None),
            raw_builder_license=("BY-NC" if index % 3 == 0 else None),
        ))
    classified = classify_records(pd.DataFrame(rows), {"gnome_r2scan_statics"})
    first = deterministic_review_sample(classified, 300, 3401)
    second = deterministic_review_sample(classified, 300, 3401)
    assert len(first) == 300
    assert first.equals(second)
    assert first.human_label.eq("").all()
    assert first.review_stratum.nunique() >= 4


def test_human_attestation_records_labels_without_changing_automated_class():
    classified = classify_records(pd.DataFrame([
        _row(transition_id="01" * 16, competitor_contextual_id="ctx-a", raw_builder_batch_id="gnome_r2scan_statics", material_builder_license="BY-NC", raw_builder_license="BY-NC"),
        _row(transition_id="02" * 16, competitor_contextual_id="ctx-u"),
    ]), {"gnome_r2scan_statics"})
    review = deterministic_review_sample(classified, 2, 3401)
    completed = apply_human_attestation(
        review, reviewer_id="PI_USER_ATTESTATION", reviewed_at_utc="2026-08-25T18:30:00+00:00",
        note="Reviewer confirmed all presented rows.",
    )
    assert completed.human_label.tolist() == completed.provenance_class.tolist()
    assert completed.reviewer_id.eq("PI_USER_ATTESTATION").all()
    assert completed.adjudication_status.eq("accepted_no_disagreement").all()
