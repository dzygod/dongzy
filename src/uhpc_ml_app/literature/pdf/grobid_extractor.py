"""GROBID 抽取器（可选，需要外部服务）。

使用前请启动 GROBID 容器::

    docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.0

然后设置环境变量 ``GROBID_URL=http://localhost:8070`` 或在构造时传入 ``base_url``。
CI **不** 启动这个服务；单测通过 ``responses`` 或 ``requests_mock`` mock HTTP 层。
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

from uhpc_ml_app.literature.pdf.base import ExtractedTable, PdfExtractor

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class GrobidExtractor(PdfExtractor):
    """调用 GROBID 的 ``/api/processFulltextDocument`` 抽取 TEI-XML，然后解析 ``<table>`` 元素。"""

    name = "grobid"
    extraction_method = "pdf_grobid_table"

    def __init__(
        self,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (base_url or os.environ.get("GROBID_URL", "http://localhost:8070")).rstrip(
            "/"
        )
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def extract(self, pdf_path: str | Path) -> list[ExtractedTable]:
        path = Path(pdf_path)
        xml = self._call_grobid(path)
        return list(self._iter_tables(xml, source_pdf=str(path)))

    # ------------------------------------------------------------------
    # HTTP & 解析（拆开便于单测 mock）
    # ------------------------------------------------------------------

    def _call_grobid(self, pdf_path: Path) -> str:
        url = f"{self._base_url}/api/processFulltextDocument"
        with pdf_path.open("rb") as fh:
            resp = self._session.post(
                url,
                files={"input": (pdf_path.name, fh, "application/pdf")},
                data={"consolidateHeader": "1", "includeRawAffiliations": "0"},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.text

    def _iter_tables(self, tei_xml: str, *, source_pdf: str) -> Any:
        root = ET.fromstring(tei_xml)
        for tbl_idx, table_el in enumerate(root.findall(".//tei:figure[@type='table']", _TEI_NS)):
            rows = _parse_tei_table(table_el)
            if not rows:
                continue
            caption_el = table_el.find(".//tei:figDesc", _TEI_NS)
            caption = "".join(caption_el.itertext()).strip() if caption_el is not None else None
            page = _int_or(table_el.get("coords", "").split(",")[0], default=tbl_idx + 1)
            yield ExtractedTable(
                source_pdf=source_pdf,
                page=page,
                rows=rows,
                caption=caption,
                confidence=3,
                extraction_method=self.extraction_method,
            )


_WHITESPACE_RE = re.compile(r"\s+")


def _parse_tei_table(table_el: ET.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_el in table_el.findall(".//tei:row", _TEI_NS):
        cells: list[str] = []
        for cell_el in row_el.findall("tei:cell", _TEI_NS):
            text = "".join(cell_el.itertext()).strip()
            cells.append(_WHITESPACE_RE.sub(" ", text))
        if any(cells):
            rows.append(cells)
    return rows


def _int_or(text: str, *, default: int) -> int:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


__all__ = ["GrobidExtractor"]
