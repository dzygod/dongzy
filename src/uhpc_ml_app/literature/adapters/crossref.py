"""Crossref 公共 API adapter。

免登录。与 OpenAlex 相比：优点是元数据规整、返回 JATS 摘要（偶尔含数字表）；缺点是中文期刊覆盖差。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import requests

from uhpc_ml_app.literature.base import LiteratureRecord, SourceAdapter
from uhpc_ml_app.literature.extraction import extract_numeric_features

_BASE_URL = "https://api.crossref.org/works"

_JATS_TAG_PATTERN = re.compile(r"<[^>]+>")


def _strip_jats(abstract: str) -> str:
    """Crossref 的 abstract 常带 `<jats:p>` 等标签，这里剥掉以方便正则。"""
    return _JATS_TAG_PATTERN.sub(" ", abstract or "").strip()


class CrossrefAdapter(SourceAdapter):
    name = "crossref"
    source = "crossref"

    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0) -> None:
        self._session = session or requests.Session()
        self._timeout = timeout

    def ingest(
        self,
        query_or_path: str,
        *,
        rows: int = 200,
        **_options: Any,
    ) -> Iterable[LiteratureRecord]:
        params: dict[str, str | int] = {"query": query_or_path, "rows": rows}
        resp = self._session.get(_BASE_URL, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("message", {}).get("items", []):
            yield self._item_to_record(item)

    def _item_to_record(self, item: dict[str, Any]) -> LiteratureRecord:
        title = (item.get("title") or [""])[0]
        journal = (item.get("container-title") or [""])[0]
        abstract = _strip_jats(item.get("abstract", ""))
        year = _parse_year(item)

        authors = item.get("author") or []
        first_author = authors[0].get("family") if authors and authors[0].get("family") else None

        doi = item.get("DOI") or None
        record_id = f"crossref:{doi or item.get('URL', '')}"

        numeric = extract_numeric_features(f"{title}. {abstract}")

        return LiteratureRecord(
            record_id=record_id,
            source=self.source,
            doi=doi,
            first_author=first_author,
            year=year,
            title=title or None,
            journal=journal or None,
            extraction_method="abstract_regex",
            confidence=1,
            raw_payload={"url": item.get("URL"), "abstract": abstract},
            **numeric,
        )


def _parse_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        obj = item.get(key, {}) or {}
        parts = obj.get("date-parts") or []
        if parts and parts[0] and parts[0][0] is not None:
            return int(parts[0][0])
    return None


__all__ = ["CrossrefAdapter"]
