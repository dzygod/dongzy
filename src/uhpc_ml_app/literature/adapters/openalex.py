"""OpenAlex 公共 API adapter。

免登录，命中率低（依赖摘要里的数字正则）。典型定位：作为 WoS/Scopus 访问受限时的
最后一道兜底。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import requests

from uhpc_ml_app.literature.base import LiteratureRecord, SourceAdapter
from uhpc_ml_app.literature.extraction import extract_numeric_features

_BASE_URL = "https://api.openalex.org/works"


def _rebuild_inverted_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, token))
    positions.sort(key=lambda kv: kv[0])
    return " ".join(tok for _, tok in positions)


class OpenAlexAdapter(SourceAdapter):
    name = "openalex"
    source = "openalex"

    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0) -> None:
        self._session = session or requests.Session()
        self._timeout = timeout

    def ingest(
        self,
        query_or_path: str,
        *,
        pages: int = 2,
        per_page: int = 100,
        **_options: Any,
    ) -> Iterable[LiteratureRecord]:
        for page in range(1, pages + 1):
            params: dict[str, str | int] = {
                "search": query_or_path,
                "per-page": per_page,
                "page": page,
                "filter": "type:article|dissertation",
            }
            resp = self._session.get(_BASE_URL, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            for work in data.get("results", []):
                yield self._work_to_record(work)

    # --- 纯函数部分，方便测试 ---------------------------------------------

    def _work_to_record(self, work: dict[str, Any]) -> LiteratureRecord:
        abstract = _rebuild_inverted_abstract(work.get("abstract_inverted_index"))
        title = work.get("display_name") or ""
        text_for_regex = f"{title}. {abstract}"
        numeric = extract_numeric_features(text_for_regex)

        doi = (work.get("doi") or "") or None
        openalex_id = work.get("id", "")
        record_id = f"openalex:{doi or openalex_id}"

        return LiteratureRecord(
            record_id=record_id,
            source=self.source,
            doi=doi,
            first_author=_first_author_of(work),
            year=work.get("publication_year"),
            title=title or None,
            journal=((work.get("primary_location") or {}).get("source") or {}).get(
                "display_name", ""
            )
            or None,
            extraction_method="abstract_regex",
            confidence=1,
            notes="",
            raw_payload={"openalex_id": openalex_id, "abstract": abstract},
            **numeric,
        )


def _first_author_of(work: dict[str, Any]) -> str | None:
    for auth in work.get("authorships", []) or []:
        display = (auth.get("author") or {}).get("display_name")
        if display:
            return display.split()[-1]
    return None


__all__ = ["OpenAlexAdapter"]
