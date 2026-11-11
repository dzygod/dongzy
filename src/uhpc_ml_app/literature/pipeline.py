"""把 adapters 产出的 :class:`LiteratureRecord` 流水化：去重 → staging DataFrame → CLEAN。

这里刻意不做数值归一化；单位换算在 adapter 侧（知道自己的来源格式）或用户手动补录时完成。
本模块只做"结构层"的合并、去重、审计、schema 校验。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera as pa

from uhpc_ml_app.literature.base import LiteratureRecord, normalize_doi
from uhpc_ml_app.literature.schemas import CLEAN_SCHEMA, STAGING_SCHEMA

# 来源优先级；小数值 = 高优先级。对齐 data_protocol.md §3.3。
SOURCE_PRIORITY: dict[str, int] = {
    "manual": 0,
    "pdf_grobid": 1,
    "pdf_pdfplumber": 2,
    "wos": 3,
    "scopus": 4,
    "cnki": 5,
    "crossref": 6,
    "openalex": 7,
}


def deduplicate(records: Iterable[LiteratureRecord]) -> tuple[list[LiteratureRecord], list[dict]]:
    """按协议去重，返回 (保留, 丢弃日志)。

    主键 = ``normalize(doi)``；退化键 = ``(first_author_lastname, year, title[:80])``。
    当存在多条同键记录时，保留 ``SOURCE_PRIORITY`` 小的那条。
    """
    best_by_key: dict[str, LiteratureRecord] = {}
    drops: list[dict] = []
    for rec in records:
        key = _dedup_key(rec)
        if key is None:
            # 没有任何可以用的键 → 保留原纪录，但记一条审计
            best_by_key[f"no_key::{rec.record_id}"] = rec
            continue
        incumbent = best_by_key.get(key)
        if incumbent is None:
            best_by_key[key] = rec
            continue
        if _priority(rec) < _priority(incumbent):
            drops.append(
                {
                    "dropped_record_id": incumbent.record_id,
                    "kept_record_id": rec.record_id,
                    "key": key,
                }
            )
            best_by_key[key] = rec
        else:
            drops.append(
                {
                    "dropped_record_id": rec.record_id,
                    "kept_record_id": incumbent.record_id,
                    "key": key,
                }
            )
    return list(best_by_key.values()), drops


def _dedup_key(rec: LiteratureRecord) -> str | None:
    if rec.doi:
        return f"doi::{normalize_doi(rec.doi)}"
    if rec.first_author and rec.year and rec.title:
        title_norm = "".join(ch for ch in rec.title.lower() if ch.isalnum())[:80]
        return f"tk::{rec.first_author.lower()}::{rec.year}::{title_norm}"
    return None


def _priority(rec: LiteratureRecord) -> int:
    return SOURCE_PRIORITY.get(rec.source, 99)


# ---------------------------------------------------------------------------


def to_staging_dataframe(records: Iterable[LiteratureRecord]) -> pd.DataFrame:
    """把 records 转成通过 ``STAGING_SCHEMA`` 校验的 DataFrame。"""
    rows: list[dict] = []
    for rec in records:
        row = asdict(rec)
        row["raw_payload_json"] = json.dumps(row.pop("raw_payload"), ensure_ascii=False)
        rows.append(row)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(STAGING_SCHEMA.columns))
    return STAGING_SCHEMA.validate(df, lazy=True)


def to_clean_dataframe(df_staging: pd.DataFrame) -> pd.DataFrame:
    """把 staging DF 过滤到 confidence ≥ 2、数值在物理上下界内的 clean 子集。

    过滤掉的行不会报错——过滤后再跑 CLEAN_SCHEMA；CLEAN_SCHEMA 只对留下的行做硬约束。
    """
    df = df_staging.copy()
    df = df[df["confidence"] >= 2]
    # 数值列超出物理上下界的行丢弃（记到 notes 可进一步扩展）
    from uhpc_ml_app.literature.schemas import _ALL_NUMERIC_BOUNDS, _NUMERIC_BOUNDS  # noqa: PLC0415

    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in _ALL_NUMERIC_BOUNDS.items():
        if col in df.columns:
            col_mask = df[col].between(lo, hi) | df[col].isna()
            mask &= col_mask
    df = df[mask]
    # clean 要求必填数值列非空；roughness_sa / 拉伸 / 抗剪为可选（至少一个目标即可）
    required_numeric = list(_NUMERIC_BOUNDS.keys())
    df = df.dropna(subset=required_numeric)
    return CLEAN_SCHEMA.validate(df, lazy=True)


# ---------------------------------------------------------------------------


def run_ingest(
    plans: list[tuple[str, str]],
    *,
    output_dir: Path,
    protocol_version: str = "0.1",
) -> dict:
    """执行一批 adapter → staging/clean 的流水线；产物写入 ``output_dir``。

    ``plans`` 的每一项是 ``(adapter_name, query_or_path)``。
    """
    from uhpc_ml_app.literature import get_adapter  # noqa: PLC0415 - 避免循环导入

    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[LiteratureRecord] = []
    per_adapter_counts: dict[str, int] = {}

    for adapter_name, query in plans:
        adapter = get_adapter(adapter_name)
        count = 0
        for rec in adapter.ingest(query):
            all_records.append(rec)
            count += 1
        per_adapter_counts[adapter_name] = count

    kept, drops = deduplicate(all_records)

    staging = to_staging_dataframe(kept)
    staging_path = output_dir / "literature_staging.parquet"
    staging.to_parquet(staging_path)

    prisma_log = _build_prisma_log(all_records, kept)
    prisma_log.to_csv(output_dir / "prisma_log.csv", index=False)

    dedup_log = pd.DataFrame(drops)
    dedup_log.to_csv(output_dir / "deduplication_log.csv", index=False)

    clean_path: str | None = None
    clean_count = 0
    try:
        clean = to_clean_dataframe(staging)
        clean_path = str(output_dir / "literature_clean.csv")
        clean.to_csv(clean_path, index=False)
        clean_count = len(clean)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        # 记录校验失败但不中断：让用户自己看 staging 做手动补录
        (output_dir / "clean_schema_error.txt").write_text(str(exc), encoding="utf-8")

    manifest = {
        "protocol_version": protocol_version,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "plans": [{"adapter": a, "query": q} for a, q in plans],
        "per_adapter_counts": per_adapter_counts,
        "total_raw": len(all_records),
        "total_kept_after_dedup": len(kept),
        "total_dropped_dedup": len(drops),
        "staging_rows": len(staging),
        "clean_rows": clean_count,
        "outputs": {
            "staging": str(staging_path),
            "prisma_log": str(output_dir / "prisma_log.csv"),
            "deduplication_log": str(output_dir / "deduplication_log.csv"),
            "clean": clean_path,
        },
    }
    manifest_path = output_dir / "ingest_run.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _build_prisma_log(
    all_records: list[LiteratureRecord],
    kept: list[LiteratureRecord],
) -> pd.DataFrame:
    kept_ids = {r.record_id for r in kept}
    rows = []
    for rec in all_records:
        rows.append(
            {
                "record_id": rec.record_id,
                "source": rec.source,
                "doi": rec.doi or "",
                "title": rec.title or "",
                "included": rec.record_id in kept_ids,
                "exclusion_reason": "" if rec.record_id in kept_ids else "duplicate",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "SOURCE_PRIORITY",
    "deduplicate",
    "run_ingest",
    "to_clean_dataframe",
    "to_staging_dataframe",
]
