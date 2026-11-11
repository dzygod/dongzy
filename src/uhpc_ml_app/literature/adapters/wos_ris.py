"""Web of Science / Engineering Village 的 RIS 文件导出解析 adapter。

**为什么是文件而不是 API**：WoS 公共 API（WoS-REST）对机构订阅绑定严格且需要 IP 白名单；
最可移植的做法是让用户在自己的机构账号下检索，然后用"导出 → RIS"下载下来。
该文件格式也兼容 Engineering Village。
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from uhpc_ml_app.literature.base import LiteratureRecord, SourceAdapter
from uhpc_ml_app.literature.extraction import extract_numeric_features


def parse_ris(text: str) -> list[dict[str, list[str]]]:
    """把 RIS 字符串拆成 list[dict]；每个 dict 的 value 是同 key 的多次出现。

    RIS 基本格式::

        TY  - JOUR
        AU  - Smith, J.
        TI  - Some title
        DO  - 10.xxx/yyy
        ER  -
    """
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    for raw in io.StringIO(text):
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        if len(line) < 6 or line[2:5] != "  -":
            # 多行字段（如 AB）的续行；追加到上一个非空 key
            if current:
                # 找到最后追加过的 key
                last_key = next(reversed(current))
                current[last_key][-1] += " " + line.strip()
            continue
        tag = line[0:2]
        value = line[6:].strip()
        if tag == "ER":
            if current:
                records.append(current)
                current = {}
            continue
        if tag == "TY":
            current = {"TY": [value]}
            continue
        current.setdefault(tag, []).append(value)
    if current:
        records.append(current)
    return records


class WosRisAdapter(SourceAdapter):
    """解析 WoS / EV 导出的 ``.ris`` 文件。

    ``ingest(path)``：``path`` 是 RIS 文件路径或原始字符串（以 ``TY  -`` 开头则按字符串处理）。
    """

    name = "wos_ris"
    source = "wos"

    def ingest(
        self,
        query_or_path: str,
        **_options: Any,
    ) -> Iterable[LiteratureRecord]:
        text = self._load(query_or_path)
        for idx, ris_rec in enumerate(parse_ris(text)):
            yield self._ris_to_record(ris_rec, idx)

    # ------------------------------------------------------------------

    @staticmethod
    def _load(query_or_path: str) -> str:
        # 允许直接传 RIS 原文（用于单测）
        if query_or_path.lstrip().startswith("TY  -"):
            return query_or_path
        return Path(query_or_path).read_text(encoding="utf-8")

    def _ris_to_record(self, ris: dict[str, list[str]], idx: int) -> LiteratureRecord:
        doi = (ris.get("DO") or [None])[0]
        title = (ris.get("TI") or ris.get("T1") or [""])[0]
        abstract = (ris.get("AB") or [""])[0]
        journal = (ris.get("JO") or ris.get("JF") or ris.get("T2") or [""])[0]
        year = _to_int((ris.get("PY") or ris.get("Y1") or [""])[0])

        authors = ris.get("AU") or ris.get("A1") or []
        first_author = authors[0].split(",")[0].strip() if authors else None

        record_id = f"wos:{doi or f'row{idx}'}"
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
            raw_payload=dict(ris),
            **numeric,
        )


def _to_int(text: str) -> int | None:
    try:
        return int(text[:4])
    except (TypeError, ValueError):
        return None


__all__ = ["WosRisAdapter", "parse_ris"]
