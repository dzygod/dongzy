"""CNKI RefWorks / NoteExpress 导出解析 adapter。

CNKI 在"导出文献"中提供若干种格式；**本 adapter 同时兼容两种常见格式**：

1. RefWorks 风格（每行 ``RT Journal Article`` / ``A1 作者`` / ``T1 题目`` / ``AB 摘要`` / ``DO DOI``）；
2. 简化 RIS 风格（复用 :class:`WosRisAdapter` 的 RIS parser）。

解析器通过检查首非空行判断格式：以 ``TY  -`` 开头走 RIS 分支，否则走 RefWorks 分支。
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from uhpc_ml_app.literature.adapters.wos_ris import parse_ris
from uhpc_ml_app.literature.base import LiteratureRecord, SourceAdapter
from uhpc_ml_app.literature.extraction import extract_numeric_features

_REFWORKS_TAGS = {
    "RT": "type",
    "A1": "author",
    "T1": "title",
    "JF": "journal",
    "YR": "year",
    "AB": "abstract",
    "DO": "doi",
    "UR": "url",
    "K1": "keywords",
}


def parse_refworks(text: str) -> list[dict[str, list[str]]]:
    """CNKI RefWorks 纯文本解析；两条记录之间用空行分隔。"""
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    for raw in io.StringIO(text):
        line = raw.rstrip("\r\n")
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if len(line) >= 3 and line[2] == " ":
            tag = line[:2]
            val = line[3:].strip()
            current.setdefault(tag, []).append(val)
        elif current:
            last_key = next(reversed(current))
            current[last_key][-1] += " " + line.strip()
    if current:
        records.append(current)
    return records


class CnkiRefWorksAdapter(SourceAdapter):
    name = "cnki_refworks"
    source = "cnki"

    def ingest(
        self,
        query_or_path: str,
        **_options: Any,
    ) -> Iterable[LiteratureRecord]:
        text = self._load(query_or_path)
        stripped = text.lstrip()
        parsed: list[dict[str, list[str]]]
        if stripped.startswith("TY  -"):
            parsed = parse_ris(text)
        else:
            parsed = parse_refworks(text)
        for idx, rec in enumerate(parsed):
            yield self._to_record(rec, idx)

    @staticmethod
    def _load(query_or_path: str) -> str:
        stripped = query_or_path.lstrip()
        if stripped.startswith(("TY  -", "RT ", "A1 ", "T1 ")):
            return query_or_path
        return Path(query_or_path).read_text(encoding="utf-8")

    def _to_record(self, rec: dict[str, list[str]], idx: int) -> LiteratureRecord:
        # RefWorks 风格字段（与 RIS 大致有对应）
        def first(keys: list[str]) -> str | None:
            for k in keys:
                vs = rec.get(k)
                if vs:
                    return vs[0]
            return None

        doi = first(["DO"])
        title = first(["T1", "TI"])
        abstract = first(["AB"]) or ""
        journal = first(["JF", "JO", "T2"])
        year_str = first(["YR", "PY", "Y1"]) or ""
        year = _to_int(year_str)

        authors = rec.get("A1") or rec.get("AU") or []
        first_author_name = authors[0] if authors else None
        first_author = first_author_name.split(",")[0].strip() if first_author_name else None

        record_id = f"cnki:{doi or f'row{idx}'}"
        numeric = extract_numeric_features(f"{title or ''}. {abstract}")

        return LiteratureRecord(
            record_id=record_id,
            source=self.source,
            doi=doi,
            first_author=first_author,
            year=year,
            title=title,
            journal=journal,
            extraction_method="abstract_regex",
            confidence=1,
            raw_payload=dict(rec),
            **numeric,
        )


def _to_int(text: str) -> int | None:
    try:
        return int(text[:4])
    except (TypeError, ValueError):
        return None


__all__ = ["CnkiRefWorksAdapter", "parse_refworks"]
