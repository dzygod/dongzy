from uhpc_ml_app.literature_crawler import (
    MODEL_FIELDNAMES,
    _count_filled_fields,
    deduplicate_records,
    extract_structured_features,
    normalize_crossref_item,
)


def test_extract_structured_features_parses_common_patterns() -> None:
    text = (
        "Interface roughness Sa = 2.35 um, sandblasted substrate. "
        "Water-binder ratio was 0.22 and fiber volume fraction was 2.0%. "
        "NC strength grade C50. Curing age 28 d. "
        "Tensile bond strength 3.1 MPa, shear bond strength 10.2 MPa."
    )
    out = extract_structured_features(text)
    assert out["roughness_sa"] == 2.35
    assert "roughness_rz" not in out
    assert out["interface_prep"] == "SB"
    assert out["water_binder_ratio"] == 0.22
    assert out["fiber_volume_ratio"] == 2.0
    assert out["nc_strength_grade"] == 50
    assert out["curing_age_days"] == 28
    assert out["tensile_bond_strength"] == 3.1
    assert out["shear_bond_strength"] == 10.2


def test_deduplicate_records_prefers_doi() -> None:
    records = [
        {"title": "A", "doi": "10.1/abc"},
        {"title": "A duplicate", "doi": "10.1/abc"},
        {"title": "B", "doi": ""},
    ]
    deduped = deduplicate_records(records)
    assert len(deduped) == 2


def test_normalize_crossref_item() -> None:
    item = {
        "title": ["UHPC-NC interface bond"],
        "DOI": "10.1000/test",
        "container-title": ["Journal of Materials"],
        "published-print": {"date-parts": [[2022, 1, 1]]},
        "URL": "https://doi.org/10.1000/test",
        "abstract": "Abstract text",
    }
    norm = normalize_crossref_item(item)
    assert norm["title"] == "UHPC-NC interface bond"
    assert norm["doi"] == "10.1000/test"
    assert norm["year"] == 2022


def test_normalize_crossref_item_handles_none_year() -> None:
    item = {
        "title": ["No year sample"],
        "DOI": "10.1000/no-year",
        "issued": {"date-parts": [[None, 1, 1]]},
    }
    norm = normalize_crossref_item(item)
    assert norm["year"] is None


def test_model_fieldnames_match_sample_schema() -> None:
    assert MODEL_FIELDNAMES == [
        "roughness_sa",
        "water_binder_ratio",
        "fiber_volume_ratio",
        "nc_strength_grade",
        "curing_age_days",
        "interface_prep",
        "tensile_bond_strength",
        "shear_bond_strength",
    ]


def test_count_filled_fields() -> None:
    row = {
        "roughness_sa": 2.1,
        "water_binder_ratio": 0.22,
        "fiber_volume_ratio": "",
        "nc_strength_grade": 50,
        "curing_age_days": None,
        "interface_prep": "SB",
        "tensile_bond_strength": 3.0,
        "shear_bond_strength": 10.0,
    }
    assert _count_filled_fields(row) == 6
