from __future__ import annotations

from uhpc_ml_app.literature.extraction import extract_numeric_features


def test_extract_numeric_features_full_match() -> None:
    text = (
        "The interface roughness was measured: Sa = 2.5 μm. "
        "The UHPC mixture used a water-binder ratio of 0.18 and a fiber volume fraction of 2.0. "
        "NC compressive strength was 50 MPa. Curing age was 28 days. "
        "Tensile bond strength reached 3.5 MPa while shear bond strength was 8.2 MPa."
    )
    feats = extract_numeric_features(text)
    assert feats["roughness_sa"] == 2.5
    assert feats["water_binder_ratio"] == 0.18
    assert feats["fiber_volume_ratio"] == 2.0
    assert feats["nc_strength_grade"] == 50.0
    assert feats["curing_age_days"] == 28.0
    assert feats["tensile_bond_strength"] == 3.5
    assert feats["shear_bond_strength"] == 8.2


def test_extract_numeric_features_c_grade_priority() -> None:
    text = "Substrate concrete grade C40 was used."
    feats = extract_numeric_features(text)
    assert feats["nc_strength_grade"] == 40.0


def test_extract_numeric_features_empty() -> None:
    feats = extract_numeric_features("")
    assert all(v is None for v in feats.values())
