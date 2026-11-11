"""Scopus "Export → CSV" 导出解析 adapter。

Scopus CSV 的默认列名格式是：
``Authors,Author full names,Author(s) ID,Title,Year,Source title,Volume,Issue,
Art. No.,Page start,Page end,Page count,Cited by,DOI,Link,Affiliations,
Authors with affiliations,Abstract,Author Keywords,...``

本 adapter 不假定列顺序，只按列名查找，缺列就返回 ``None``。
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from uhpc_ml_app.literature.base import LiteratureRecord, SourceAdapter
from uhpc_ml_app.literature.extraction import extract_numeric_features


class ScopusCsvAdapter(SourceAdapter):
    name = "scopus_csv"
    source = "scopus"

    def ingest(
        self,
        query_or_path: str,
        **_options: Any,
    ) -> Iterable[LiteratureRecord]:
        text = self._load(query_or_path)
        reader = csv.DictReader(io.StringIO(text))
        for idx, row in enumerate(reader):
            yield self._row_to_record(row, idx)

    @staticmethod
    def _load(query_or_path: str) -> str:
        # 允许直接传 CSV 字符串用于单测
        if "," in query_or_path and "\n" in query_or_path and not query_or_path.endswith(".csv"):
            return query_or_path
        return Path(query_or_path).read_text(encoding="utf-8-sig")

    def _row_to_record(self, row: dict[str, str], idx: int) -> LiteratureRecord:
        # Scopus 的列名大小写稳定；这里做一次小的标准化以防不同区域差异。
        normalized = {k.strip().lower(): (v or "").strip() for k, v in row.items()}

        doi = normalized.get("doi") or None
        title = normalized.get("title") or None
        abstract = normalized.get("abstract") or ""
        journal = normalized.get("source title") or None
        year = _to_int(normalized.get("year", ""))

        authors = normalized.get("authors") or normalized.get("author full names") or ""
        first_author = _first_author_from_scopus(authors)

        record_id = f"scopus:{doi or f'row{idx}'}"
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
            raw_payload=dict(row),
            **numeric,
        )


def _first_author_from_scopus(authors_field: str) -> str | None:
    """Scopus 作者字段例：``Smith J.; Wang L.``；取第一个，并返回姓氏。"""
    if not authors_field:
        return None
    first = authors_field.split(";")[0].strip()
    return first.split(",")[0].split()[0] if first else None


def _to_int(text: str) -> int | None:
    try:
        return int(text[:4])
    except (TypeError, ValueError):
        return None


__all__ = ["ScopusCsvAdapter"]
