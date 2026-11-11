from __future__ import annotations

from typing import Any

import pytest

from uhpc_ml_app.literature.pdf import GrobidExtractor, PdfPlumberExtractor
from uhpc_ml_app.literature.pdf.base import ExtractedTable


def test_pdfplumber_extractor_raises_if_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import uhpc_ml_app.literature.pdf.pdfplumber_extractor as mod

    monkeypatch.setattr(mod, "_load_pdfplumber", lambda: None)
    extractor = PdfPlumberExtractor()
    with pytest.raises(RuntimeError, match="pdfplumber is not installed"):
        extractor.extract("whatever.pdf")


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:  # pragma: no cover - no-op
        return None


class _FakeSession:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_url: str | None = None

    def post(self, url: str, **_: Any) -> _FakeResp:
        self.last_url = url
        return _FakeResp(self._text)


_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body>
    <figure type="table" xml:id="tab1" coords="3,140,200,300,400">
      <figDesc>Bond strength vs roughness</figDesc>
      <table>
        <row><cell>Ra (um)</cell><cell>Tensile (MPa)</cell></row>
        <row><cell>2.5</cell><cell>3.2</cell></row>
        <row><cell>5.0</cell><cell>4.1</cell></row>
      </table>
    </figure>
  </body></text>
</TEI>"""


def test_grobid_extractor_parses_tei_xml(tmp_path: Any) -> None:
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%fake")
    session = _FakeSession(_TEI_XML)
    extractor = GrobidExtractor(base_url="http://grobid.local", session=session)  # type: ignore[arg-type]
    tables = extractor.extract(fake_pdf)

    assert session.last_url == "http://grobid.local/api/processFulltextDocument"
    assert len(tables) == 1
    table = tables[0]
    assert isinstance(table, ExtractedTable)
    assert table.rows == [
        ["Ra (um)", "Tensile (MPa)"],
        ["2.5", "3.2"],
        ["5.0", "4.1"],
    ]
    assert table.caption == "Bond strength vs roughness"
    assert table.confidence == 3
    assert table.extraction_method == "pdf_grobid_table"
