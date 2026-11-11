"""文献宽表的 pandera schema。

与 ``src/uhpc_ml_app/schemas.py`` 的 INPUT/CLEAN 两段式思路一致，本模块额外扩展了
审计字段（``source`` / ``doi`` / ``extraction_method`` / ``confidence`` / ``raw_payload_json``）。

``STAGING_SCHEMA`` 是"原始入库"阶段的 schema，容忍 NaN 与低置信度；
``CLEAN_SCHEMA`` 是进入建模管道前的最终 schema，与 ``uhpc_ml_app.schemas.CLEAN_SCHEMA``
在数值列上的物理上下界保持一致，额外要求 ``confidence >= 2`` 且必须有 DOI 或 first_author+year。

自 v0.3 起，界面粗糙度改用 ``interface_prep``（分类）或 ``roughness_sa``（三维数值）表达，
二者可二选一；输出目标（拉伸/抗剪）允许只填其一。因此数值列拆分为：

- ``_NUMERIC_BOUNDS``：必填数值（clean 阶段必须非空）；
- ``_OPTIONAL_NUMERIC_BOUNDS``：可选数值（clean 阶段可空）。
"""

from __future__ import annotations

from pandera import Check, Column, DataFrameSchema

from uhpc_ml_app.literature.base import EXTRACTION_METHODS, SOURCES

# --- 必填数值列的物理上下界（clean 阶段非空）---
_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "water_binder_ratio": (0.08, 0.60),
    "fiber_volume_ratio": (0.0, 6.0),
    "nc_strength_grade": (10.0, 120.0),
    "curing_age_days": (0.5, 400.0),
}

# --- 可选数值列的物理上下界（clean 阶段可空）---
_OPTIONAL_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "roughness_sa": (0.0, 50.0),
    "tensile_bond_strength": (0.0, 20.0),
    "shear_bond_strength": (0.0, 40.0),
}

# 全部数值列（bounds 过滤用）
_ALL_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    **_NUMERIC_BOUNDS,
    **_OPTIONAL_NUMERIC_BOUNDS,
}

TARGET_COLUMNS = ["tensile_bond_strength", "shear_bond_strength"]


def _staging_numeric(name: str) -> Column:
    """staging 阶段：允许 NaN，但若非 NaN 必须非负。"""
    return Column(
        float,
        checks=[Check.greater_than_or_equal_to(0.0)],
        nullable=True,
        coerce=True,
        description=f"{name} (nullable in staging; physical bounds enforced in CLEAN_SCHEMA)",
    )


def _clean_numeric(name: str, *, nullable: bool) -> Column:
    lo, hi = _ALL_NUMERIC_BOUNDS[name]
    return Column(
        float,
        checks=[Check.in_range(lo, hi, include_min=True, include_max=True)],
        nullable=nullable,
        coerce=True,
        description=f"{name} ∈ [{lo}, {hi}]" + (" (nullable)" if nullable else ""),
    )


# 审计字段
_AUDIT_COLS_STAGING: dict[str, Column] = {
    "record_id": Column(str, nullable=False, unique=True),
    "source": Column(str, checks=Check.isin(SOURCES), nullable=False),
    "doi": Column(str, nullable=True),
    "first_author": Column(str, nullable=True),
    "year": Column("Int64", nullable=True, coerce=True),
    "title": Column(str, nullable=True),
    "journal": Column(str, nullable=True),
    "specimen_type": Column(str, nullable=True),
    "interface_prep": Column(str, nullable=True),
    "extraction_method": Column(str, checks=Check.isin(EXTRACTION_METHODS), nullable=False),
    "confidence": Column(
        "Int64",
        checks=Check.in_range(0, 3, include_min=True, include_max=True),
        nullable=False,
        coerce=True,
    ),
    "notes": Column(str, nullable=True),
    "raw_payload_json": Column(str, nullable=True),
}


_AUDIT_COLS_CLEAN: dict[str, Column] = {
    **_AUDIT_COLS_STAGING,
    "confidence": Column(
        "Int64",
        checks=Check.greater_than_or_equal_to(2),
        nullable=False,
        coerce=True,
        description="进入建模的记录必须 confidence ≥ 2",
    ),
}


STAGING_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={
        **_AUDIT_COLS_STAGING,
        **{name: _staging_numeric(name) for name in _ALL_NUMERIC_BOUNDS},
    },
    strict="filter",
    coerce=True,
    ordered=False,
)


CLEAN_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={
        **_AUDIT_COLS_CLEAN,
        **{name: _clean_numeric(name, nullable=False) for name in _NUMERIC_BOUNDS},
        **{name: _clean_numeric(name, nullable=True) for name in _OPTIONAL_NUMERIC_BOUNDS},
    },
    strict="filter",
    coerce=True,
    ordered=False,
    checks=[
        Check(
            lambda df: df["doi"].notna() | (df["first_author"].notna() & df["year"].notna()),
            name="doi_or_author_year_present",
            error="每条 clean 记录必须至少提供 DOI 或 (first_author + year) 作为溯源键",
        ),
        Check(
            lambda df: bool(df[TARGET_COLUMNS].notna().any(axis=1).all()),
            name="each_row_has_at_least_one_target",
            error="每条 clean 记录必须至少提供 tensile_bond_strength 或 shear_bond_strength 之一",
        ),
    ],
)


__all__ = ["CLEAN_SCHEMA", "STAGING_SCHEMA"]
