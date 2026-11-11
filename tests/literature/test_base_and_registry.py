from __future__ import annotations

import pytest

from uhpc_ml_app.literature import LiteratureRecord, get_adapter, list_adapters
from uhpc_ml_app.literature.base import normalize_doi


def test_registry_has_five_builtin_adapters() -> None:
    names = set(list_adapters())
    assert {
        "openalex",
        "crossref",
        "wos_ris",
        "scopus_csv",
        "cnki_refworks",
    }.issubset(names)


def test_get_adapter_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_adapter("definitely-not-an-adapter")


def test_normalize_doi_strips_prefix_and_case() -> None:
    assert (
        normalize_doi("HTTPS://DOI.ORG/10.1016/J.Cemconres.2023.107123")
        == "10.1016/j.cemconres.2023.107123"
    )
    assert normalize_doi("doi:10.1016/X") == "10.1016/x"
    assert normalize_doi("  10.1016/y  ") == "10.1016/y"


def test_literature_record_validates_enums() -> None:
    rec = LiteratureRecord(record_id="x:1", source="openalex")
    assert rec.extraction_method == "abstract_regex"
    assert rec.confidence == 1

    with pytest.raises(ValueError, match="unknown source"):
        LiteratureRecord(record_id="x:2", source="twitter")
    with pytest.raises(ValueError, match="unknown extraction_method"):
        LiteratureRecord(record_id="x:3", source="openalex", extraction_method="ouija")
    with pytest.raises(ValueError, match="confidence must be in 0..3"):
        LiteratureRecord(record_id="x:4", source="openalex", confidence=5)


def test_literature_record_normalizes_doi() -> None:
    rec = LiteratureRecord(
        record_id="x:5",
        source="crossref",
        doi="https://doi.org/10.1000/Abc",
    )
    assert rec.doi == "10.1000/abc"
