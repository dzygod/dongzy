"""对 OpenAlex / Crossref adapter 的单测；HTTP 层用 fake session mock 掉。"""

from __future__ import annotations

from typing import Any

from uhpc_ml_app.literature.adapters.crossref import CrossrefAdapter
from uhpc_ml_app.literature.adapters.openalex import OpenAlexAdapter


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - no-op
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _FakeResponse(self._payloads.pop(0))


def test_openalex_adapter_builds_records_from_inverted_abstract() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "display_name": "UHPC-NC interfacial bond study",
                "doi": "https://doi.org/10.1/abc",
                "publication_year": 2022,
                "primary_location": {"source": {"display_name": "Cem. Conc. Res."}},
                "authorships": [{"author": {"display_name": "Jane Doe"}}],
                "abstract_inverted_index": {
                    "Sa": [0],
                    "=": [1],
                    "3.0": [2],
                    "tensile": [3],
                    "bond": [4],
                    "strength": [5],
                    "was": [6],
                    "4.2": [7],
                    "MPa.": [8],
                },
            }
        ]
    }
    session = _FakeSession([payload])
    adapter = OpenAlexAdapter(session=session)  # type: ignore[arg-type]
    records = list(adapter.ingest("uhpc", pages=1))
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "openalex"
    assert rec.doi == "10.1/abc"
    assert rec.first_author == "Doe"
    assert rec.year == 2022
    assert rec.roughness_sa == 3.0
    # tensile bond strength was 4.2 → 捕获
    assert rec.tensile_bond_strength == 4.2
    assert session.calls[0]["url"].endswith("/works")


def test_openalex_adapter_handles_null_primary_location_source() -> None:
    # OpenAlex 经常把 primary_location.source 设为 null（期刊未识别），适配器不能崩
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W999",
                "display_name": "Unknown journal paper",
                "doi": None,
                "publication_year": 2020,
                "primary_location": {"source": None},
                "authorships": [],
                "abstract_inverted_index": {"Sa": [0], "=": [1], "2.0": [2]},
            }
        ]
    }
    session = _FakeSession([payload])
    adapter = OpenAlexAdapter(session=session)  # type: ignore[arg-type]
    records = list(adapter.ingest("uhpc", pages=1))
    assert len(records) == 1
    assert records[0].journal is None


def test_crossref_adapter_strips_jats_and_parses_numbers() -> None:
    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.1/def",
                    "title": ["UHPC repair roughness study"],
                    "container-title": ["Construction and Building Materials"],
                    "issued": {"date-parts": [[2021]]},
                    "author": [{"family": "Wang", "given": "Li"}],
                    "URL": "https://doi.org/10.1/def",
                    "abstract": "<jats:p>Sa = 7.0 and shear bond strength was 11.3 MPa.</jats:p>",
                }
            ]
        }
    }
    session = _FakeSession([payload])
    adapter = CrossrefAdapter(session=session)  # type: ignore[arg-type]
    records = list(adapter.ingest("uhpc"))
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "crossref"
    assert rec.doi == "10.1/def"
    assert rec.first_author == "Wang"
    assert rec.year == 2021
    assert rec.roughness_sa == 7.0
    assert rec.shear_bond_strength == 11.3
