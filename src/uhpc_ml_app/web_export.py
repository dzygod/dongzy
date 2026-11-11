"""Streamlit 界面读取 `run_pipeline` 产物用的辅助函数。

PR#7 把原先只提供 `build_results_zip` 的小模块扩成**读取层**：所有解析
`outputs/<run>` 目录的逻辑都收敛到这里，Streamlit 侧只负责把结果渲染到组件上。
这样做的好处：

- 读取逻辑可单测（`tests/test_web_export.py`），Streamlit 本身不需要 headless 测；
- 一致的容错：缺失产物返回空对象（而不是抛异常），让 UI 能优雅降级到"该产物尚未生成"；
- 产物演进时只改这一处。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from uhpc_ml_app.config import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Tabular artifacts
# ---------------------------------------------------------------------------


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_metrics_summary(output_dir: str | Path) -> pd.DataFrame:
    """`metrics_summary.csv`：每模型 overall train/test + CV mean/95% CI。"""
    return _read_csv_optional(Path(output_dir) / "metrics_summary.csv")


def load_per_target_metrics(output_dir: str | Path) -> pd.DataFrame:
    """`metrics_per_target.csv`：每模型 × 每目标 × 训练/测试 的 R²/RMSE/MAE。"""
    return _read_csv_optional(Path(output_dir) / "metrics_per_target.csv")


def load_cv_metrics(output_dir: str | Path) -> pd.DataFrame:
    """`cv_metrics.csv`：RepeatedKFold 的 R² 均值 + 95% CI 上下界。"""
    return _read_csv_optional(Path(output_dir) / "cv_metrics.csv")


def load_baseline_metrics(output_dir: str | Path) -> pd.DataFrame:
    """`baseline_metrics.csv`：三基线（mean / linear / τ=k√fc）的测试集指标 + CV 均值。"""
    return _read_csv_optional(Path(output_dir) / "baseline_metrics.csv")


def load_permutation_importance(output_dir: str | Path) -> pd.DataFrame:
    """`permutation_importance.csv`：跨模型统一的特征重要性（R² 下降量）。"""
    return _read_csv_optional(Path(output_dir) / "permutation_importance.csv")


def load_conditional_recommendations(output_dir: str | Path) -> pd.DataFrame:
    """`roughness_conditional_recommendations.csv`：按 w/b × 养护龄期 3×3 桶的条件化推荐。

    若全局均无条件化结果（如极小样本），文件可能不存在，此时返回空 DataFrame。
    """
    return _read_csv_optional(Path(output_dir) / "roughness_conditional_recommendations.csv")


# ---------------------------------------------------------------------------
# JSON artifacts
# ---------------------------------------------------------------------------


def _read_json_optional(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_best_model_info(output_dir: str | Path) -> dict:
    """`best_model_info.json`：最优模型、model_selection_metric、
    shap_importance_by_target、roughness_sa_recommendation 等顶层摘要。"""
    return _read_json_optional(Path(output_dir) / "best_model_info.json")


def load_roughness_recommendations(output_dir: str | Path) -> dict:
    """`roughness_recommendations.json`：每个 (target, feature) 的全局 + 条件化推荐，
    以及 `explanation_method` 与 `failed_targets`。"""
    return _read_json_optional(Path(output_dir) / "roughness_recommendations.json")


# ---------------------------------------------------------------------------
# Plot discovery — 用 glob 而不是硬编码文件名，以便未来新增特征/目标时自动收录
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotIndex:
    """按"图表类别"分组的路径索引，供 UI 按 tab 渲染。"""

    scatter: dict[str, dict[str, Path]]  # scatter[model][target] -> path
    shap_summary: dict[str, Path]  # shap_summary[target] -> path
    shap_dependence: dict[str, dict[str, Path]]  # shap_dependence[target][feature] -> path
    permutation_importance: dict[str, Path]  # permutation_importance[model] -> path
    other: list[Path]  # 未归类图，兜底显示


def discover_plots(output_dir: str | Path) -> PlotIndex:
    """扫描 `output_dir/*.png` 并按命名约定分类到 :class:`PlotIndex`。

    命名约定（来自 `reporting.py` / `shap_analysis.py`）：

    - 预测散点：``scatter_{model}_{target}.png``
    - SHAP 总结：``shap_summary_{target}.png``
    - SHAP 依赖：``shap_dependence_{target}_{feature}.png``
    - Permutation Importance：``permutation_importance_{model}.png``
    """
    out = Path(output_dir)
    scatter: dict[str, dict[str, Path]] = {}
    shap_summary: dict[str, Path] = {}
    shap_dependence: dict[str, dict[str, Path]] = {}
    permutation_importance: dict[str, Path] = {}
    other: list[Path] = []

    for path in sorted(out.glob("*.png")):
        stem = path.stem
        if stem.startswith("scatter_"):
            rest = stem[len("scatter_") :]
            model, target = _split_model_target(rest)
            if model and target:
                scatter.setdefault(model, {})[target] = path
                continue
        if stem.startswith("shap_summary_"):
            target = stem[len("shap_summary_") :]
            if target:
                shap_summary[target] = path
                continue
        if stem.startswith("shap_dependence_"):
            rest = stem[len("shap_dependence_") :]
            target, feature = _split_target_feature(rest)
            if target and feature:
                shap_dependence.setdefault(target, {})[feature] = path
                continue
        if stem.startswith("permutation_importance_"):
            model = stem[len("permutation_importance_") :]
            if model:
                permutation_importance[model] = path
                continue
        other.append(path)

    return PlotIndex(
        scatter=scatter,
        shap_summary=shap_summary,
        shap_dependence=shap_dependence,
        permutation_importance=permutation_importance,
        other=other,
    )


# ``{model}_{target}`` 的拆分需要容忍 model 名（如 "random_forest"）或 target 名
# （如 "tensile_bond_strength"）里本身就含下划线。用已知目标名列表来锚定后缀。
_KNOWN_TARGETS: tuple[str, ...] = ("tensile_bond_strength", "shear_bond_strength")


def _split_model_target(token: str) -> tuple[str, str]:
    for target in _KNOWN_TARGETS:
        suffix = "_" + target
        if token.endswith(suffix):
            return token[: -len(suffix)], target
    return "", ""


def _split_target_feature(token: str) -> tuple[str, str]:
    for target in _KNOWN_TARGETS:
        prefix = target + "_"
        if token.startswith(prefix):
            feat = token[len(prefix) :]
            if feat in FEATURE_COLUMNS:
                return target, feat
    return "", ""


# ---------------------------------------------------------------------------
# Bundle export
# ---------------------------------------------------------------------------


def build_results_zip(output_dir: str | Path) -> bytes:
    """把 `output_dir` 下全部一级产物打成 ZIP 供一键下载。"""
    output_path = Path(output_dir)
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as zf:
        for file_path in sorted(output_path.glob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    return buffer.getvalue()


__all__ = [
    "PlotIndex",
    "build_results_zip",
    "discover_plots",
    "load_baseline_metrics",
    "load_best_model_info",
    "load_conditional_recommendations",
    "load_cv_metrics",
    "load_metrics_summary",
    "load_per_target_metrics",
    "load_permutation_importance",
    "load_roughness_recommendations",
]
