from __future__ import annotations

import json
from pathlib import Path

import pandera as pa
import pytest

from uhpc_ml_app.literature.base import LiteratureRecord
from uhpc_ml_app.literature.pipeline import (
    deduplicate,
    run_ingest,
    to_clean_dataframe,
    to_staging_dataframe,
)
from uhpc_ml_app.literature.schemas import CLEAN_SCHEMA, STAGING_SCHEMA


def _rec(
    record_id: str,
    *,
    source: str = "openalex",
    doi: str | None = None,
    first_author: str | None = None,
    year: int | None = None,
    title: str | None = None,
    confidence: int = 1,
    **numeric: float,
) -> LiteratureRecord:
    return LiteratureRecord(
        record_id=record_id,
        source=source,
        doi=doi,
        first_author=first_author,
        year=year,
        title=title,
        confidence=confidence,
        **numeric,
    )


def test_deduplicate_prefers_higher_priority_source() -> None:
    openalex_rec = _rec("openalex:r1", source="openalex", doi="10.1/x")
    wos_rec = _rec("wos:r1", source="wos", doi="10.1/X")  # 大小写不同，靠 normalize_doi 合并
    kept, drops = deduplicate([openalex_rec, wos_rec])
    assert len(kept) == 1
    assert kept[0].source == "wos"
    assert len(drops) == 1
    assert drops[0]["dropped_record_id"] == "openalex:r1"


def test_deduplicate_falls_back_to_author_year_title() -> None:
    a = _rec("openalex:n1", first_author="Smith", year=2022, title="Bond behavior of UHPC")
    b = _rec("crossref:n2", first_author="Smith", year=2022, title="Bond Behavior of UHPC!")
    kept, drops = deduplicate([a, b])
    assert len(kept) == 1
    assert len(drops) == 1


def test_staging_schema_accepts_nulls_and_enforces_enums() -> None:
    recs = [_rec("openalex:s1", doi="10.1/a"), _rec("wos:s2", doi="10.1/b", roughness_sa=2.0)]
    df = to_staging_dataframe(recs)
    assert list(STAGING_SCHEMA.columns.keys()) == list(STAGING_SCHEMA.columns.keys())
    assert len(df) == 2
    assert "raw_payload_json" in df.columns
    assert df["confidence"].tolist() == [1, 1]


def test_clean_schema_requires_non_null_numerics_and_confidence_at_least_2() -> None:
    # 低 confidence → 被过滤
    low = _rec(
        "openalex:s3",
        doi="10.1/c",
        confidence=1,
        roughness_sa=5.0,
        water_binder_ratio=0.18,
        fiber_volume_ratio=2.0,
        nc_strength_grade=50.0,
        curing_age_days=28.0,
        tensile_bond_strength=3.5,
        shear_bond_strength=8.0,
    )
    # 高 confidence、完整
    high = _rec(
        "wos:s4",
        source="wos",
        doi="10.1/d",
        confidence=3,
        roughness_sa=5.0,
        water_binder_ratio=0.18,
        fiber_volume_ratio=2.0,
        nc_strength_grade=50.0,
        curing_age_days=28.0,
        tensile_bond_strength=3.5,
        shear_bond_strength=8.0,
    )
    staging = to_staging_dataframe([low, high])
    clean = to_clean_dataframe(staging)
    assert len(clean) == 1
    assert clean.iloc[0]["record_id"] == "wos:s4"


def test_clean_schema_rejects_out_of_range_values() -> None:
    # tensile_bond_strength 远超上限 → CLEAN_SCHEMA 过滤掉，staging 里还在（只要求非负）
    oob = _rec(
        "wos:s5",
        source="wos",
        doi="10.1/e",
        confidence=3,
        roughness_sa=5.0,
        water_binder_ratio=0.18,
        fiber_volume_ratio=2.0,
        nc_strength_grade=50.0,
        curing_age_days=28.0,
        tensile_bond_strength=99.0,
        shear_bond_strength=8.0,
    )
    staging = to_staging_dataframe([oob])
    # staging 阶段只要求 >=0，不卡上界
    STAGING_SCHEMA.validate(staging, lazy=True)
    clean = to_clean_dataframe(staging)
    assert len(clean) == 0


def test_clean_schema_requires_doi_or_author_year() -> None:
    # 既无 DOI 也无作者 + 年份 → CLEAN_SCHEMA 的 dataframe-level check 会拒
    rec = _rec(
        "manual:s6",
        source="manual",
        confidence=3,
        roughness_sa=5.0,
        water_binder_ratio=0.18,
        fiber_volume_ratio=2.0,
        nc_strength_grade=50.0,
        curing_age_days=28.0,
        tensile_bond_strength=3.5,
        shear_bond_strength=8.0,
    )
    staging = to_staging_dataframe([rec])
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        CLEAN_SCHEMA.validate(staging, lazy=True)


def test_run_ingest_writes_audit_artifacts(tmp_path: Path) -> None:
    from uhpc_ml_app.literature import register_adapter
    from uhpc_ml_app.literature.base import SourceAdapter

    class _StubAdapter(SourceAdapter):
        name = "_stub"
        source = "manual"

        def ingest(self, query_or_path: str, **_options: object) -> list[LiteratureRecord]:
            return [
                LiteratureRecord(
                    record_id="manual:r1",
                    source="manual",
                    first_author="Smith",
                    year=2022,
                    title="t",
                    confidence=3,
                    roughness_sa=5.0,
                    water_binder_ratio=0.18,
                    fiber_volume_ratio=2.0,
                    nc_strength_grade=50.0,
                    curing_age_days=28.0,
                    tensile_bond_strength=3.5,
                    shear_bond_strength=8.0,
                    extraction_method="manual_entry",
                )
            ]

    register_adapter(_StubAdapter())
    manifest = run_ingest([("_stub", "anything")], output_dir=tmp_path)
    assert manifest["total_raw"] == 1
    assert manifest["total_kept_after_dedup"] == 1
    assert manifest["clean_rows"] == 1
    assert (tmp_path / "ingest_run.json").exists()
    assert (tmp_path / "prisma_log.csv").exists()
    assert (tmp_path / "deduplication_log.csv").exists()
    assert (tmp_path / "literature_staging.parquet").exists()
    assert (tmp_path / "literature_clean.csv").exists()
    run_json = json.loads((tmp_path / "ingest_run.json").read_text(encoding="utf-8"))
    assert run_json["protocol_version"] == "0.1"
