"""数据读取与清洗流水线：规范化 → 中位数填补 → IQR → interface_prep one-hot。"""

from pathlib import Path

import pandas as pd

from uhpc_ml_app.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    INTERFACE_PREP_VALUES,
    NUMERIC_FEATURE_COLUMNS,
    SPARSE_FEATURE_COLUMNS,
    SPARSE_MISSING,
    SPECIMEN_TYPE_VALUES,
    TARGET_COLUMNS,
)
from uhpc_ml_app.schemas import CLEAN_SCHEMA, INPUT_SCHEMA

# 视为"未提供界面处理方式"的字符串（大小写不敏感，输入先统一为大写再匹配）
_INTERFACE_PREP_MISSING: tuple[str, ...] = (
    "",
    "NAN",
    "NONE",
    "NULL",
    "UNKNOWN",
    "N/A",
    "NA",
    "-",
)


def _normalize_interface_prep(series: pd.Series) -> pd.Series:
    """把界面处理方式统一为大写编号（如 sb → SB）、去空白；缺失/占位符归一为 None。"""
    s = series.astype(str).str.strip().str.upper()
    return s.replace(list(_INTERFACE_PREP_MISSING), None)


def _clip_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """使用 IQR 规则逐列剔除异常值（仅作用于连续数值特征）。

    退化特征（IQR≈0，即低基数/近似常数，如 ``fiber_volume_ratio`` 绝大多数为 2%）
    跳过裁剪，否则会把所有非众数取值（0/1/2.2% 纤维）当作离群误删。
    """
    clipped = df.copy()
    for col in columns:
        q1 = clipped[col].quantile(0.25)
        q3 = clipped[col].quantile(0.75)
        iqr = q3 - q1
        if iqr < 1e-12:  # 退化特征：IQR=0，跳过裁剪
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        clipped = clipped[(clipped[col] >= low) & (clipped[col] <= high)]
    return clipped


def _onehot_interface_prep(df: pd.DataFrame) -> pd.DataFrame:
    """把 ``interface_prep`` 展开为 ``interface_prep__<value>`` 0/1 列。

    - 命中 ``INTERFACE_PREP_VALUES`` 的值 → 对应列置 1；
    - 其余值 / 缺失 → 全 0（作为 reference 基准，表示"未提供/其他处理方式"）。
    """
    prep = df["interface_prep"] if "interface_prep" in df.columns else pd.Series(dtype=str)
    out = df.drop(columns=["interface_prep"], errors="ignore").copy()
    for value in INTERFACE_PREP_VALUES:
        col = f"interface_prep__{value}"
        out[col] = (prep == value).astype(int)
    return out


def _normalize_specimen_type(series: pd.Series) -> pd.Series:
    """把试件/试验类型统一为小写、去空白；缺失/占位符归一为 None。"""
    s = series.astype(str).str.strip().str.lower()
    return s.replace([m.lower() for m in _INTERFACE_PREP_MISSING], None)


def _onehot_specimen_type(df: pd.DataFrame) -> pd.DataFrame:
    """把 ``specimen_type`` 展开为 ``specimen_type__<value>`` 0/1 列。

    - 命中 ``SPECIMEN_TYPE_VALUES`` 的值 → 对应列置 1；
    - 其余值 / 缺失 → 全 0（作为 reference 基准）。
    """
    st = df["specimen_type"] if "specimen_type" in df.columns else pd.Series(dtype=str)
    out = df.drop(columns=["specimen_type"], errors="ignore").copy()
    for value in SPECIMEN_TYPE_VALUES:
        col = f"specimen_type__{value}"
        out[col] = (st == value).astype(int)
    return out


def load_and_clean_data(csv_path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """读取并清洗文献整合 CSV 数据。

    与旧版差异：
    - 界面粗糙度只用 ``interface_prep``（分类）表征；``roughness_sa`` 不再作为建模特征
      （列保留于宽表但清洗时丢弃，因其在已纳入文献中近似为常数）；
    - 新增 ``specimen_type``（试件/试验类型）分类特征，与 ``interface_prep`` 一并 one-hot；
    - 连续数值特征缺失时用中位数填补；``interface_prep`` / ``specimen_type`` 缺失归入基准类别；
    - 输出目标（拉伸/抗剪）允许缺失，但至少填一项；不做目标中位数填补，保留 NaN 以便按目标独立建模。
    """
    df = pd.read_csv(csv_path)
    required = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS + TARGET_COLUMNS

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    # 稀疏特征列为可选：CSV 提供则取，缺失则补 NaN（后续统一填哨兵值，向后兼容无该列的旧 CSV）
    sparse_cols = [c for c in SPARSE_FEATURE_COLUMNS if c in df.columns]
    work = df[required + sparse_cols].copy()
    for c in SPARSE_FEATURE_COLUMNS:
        if c not in work.columns:
            work[c] = float("nan")
    for col in NUMERIC_FEATURE_COLUMNS + SPARSE_FEATURE_COLUMNS + TARGET_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["interface_prep"] = _normalize_interface_prep(work["interface_prep"])
    work["specimen_type"] = _normalize_specimen_type(work["specimen_type"])

    if validate:
        work = INPUT_SCHEMA.validate(work, lazy=True)

    work = work.dropna(axis=0, how="all")
    # 至少有一个输出标签
    work = work[work[TARGET_COLUMNS].notna().any(axis=1)]
    # 连续输入特征用中位数填补后仍缺则删行
    work[NUMERIC_FEATURE_COLUMNS] = work[NUMERIC_FEATURE_COLUMNS].fillna(
        work[NUMERIC_FEATURE_COLUMNS].median(numeric_only=True)
    )
    work = work.dropna(subset=NUMERIC_FEATURE_COLUMNS)
    work = _clip_outliers_iqr(work, NUMERIC_FEATURE_COLUMNS)
    # 稀疏特征：缺失填哨兵值（不中位数填补、不 IQR 裁剪），供树模型做缺失指示切分
    work[SPARSE_FEATURE_COLUMNS] = work[SPARSE_FEATURE_COLUMNS].fillna(SPARSE_MISSING)
    work = work.reset_index(drop=True)

    if work.empty:
        raise ValueError("数据清洗后为空，请检查输入数据质量。")

    # interface_prep / specimen_type → one-hot（缺失/其他 → 全 0 基准）
    work = _onehot_interface_prep(work)
    work = _onehot_specimen_type(work)

    if validate:
        work = CLEAN_SCHEMA.validate(work, lazy=True)
    return work


__all__ = ["load_and_clean_data"]
