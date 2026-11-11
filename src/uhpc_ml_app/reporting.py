"""结果输出模块：指标表、预测散点图、特征重要性、SHAP 解释与最优模型摘要。

PR#2 起本模块的对外 API 保持**向下兼容**：

- ``save_metrics`` / ``save_prediction_scatter`` / ``run_shap_analysis`` 继续
  存在、继续输出 ``metrics_summary.csv`` 与 ``best_model_info.json``；
- 新增 ``save_per_target_metrics`` / ``save_cv_metrics`` /
  ``save_permutation_importance`` / ``save_baseline_metrics``，按目标与基线
  拆分指标，便于论文中写"按目标的 R²/RMSE/MAE 与 Repeated K-Fold 95% CI"表格。

PR#3 将 ``run_shap_analysis`` 改为薄壳，委托至
``uhpc_ml_app.shap_analysis.run_dual_target_shap``（双目标 + 依赖图驱动阈值）。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS

# ---------------------------------------------------------------------------
# Metrics tables
# ---------------------------------------------------------------------------


def save_metrics(result: dict, output_dir: Path) -> None:
    """导出 ``metrics_summary.csv``：每模型的 overall train/test 指标。

    为了兼容现有 Streamlit UI 与论文表格直接拷贝使用，列顺序与原版保持一致，
    同时补充 best_params 的 JSON 字符串便于审查。
    """
    rows = []
    cv_metrics = result.get("cv_metrics", {})
    for model, metric in result["metrics"].items():
        row = {
            "model": model,
            "train_r2": metric["train"]["r2"],
            "train_rmse": metric["train"]["rmse"],
            "train_mae": metric["train"]["mae"],
            "test_r2": metric["test"]["r2"],
            "test_rmse": metric["test"]["rmse"],
            "test_mae": metric["test"]["mae"],
        }
        cv = cv_metrics.get(model, {})
        row["cv_r2_mean"] = cv.get("r2_mean", float("nan"))
        row["cv_r2_ci95_low"] = cv.get("r2_ci95_low", float("nan"))
        row["cv_r2_ci95_high"] = cv.get("r2_ci95_high", float("nan"))
        row["best_params"] = json.dumps(metric.get("best_params", {}), ensure_ascii=False)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)


def save_per_target_metrics(result: dict, output_dir: Path) -> None:
    """导出 ``metrics_per_target.csv``（长表）：每模型 × 每目标 × 训练/测试 的指标。

    论文正文/附表中"拉伸粘结强度 R²、抗剪粘结强度 R²"需要分列呈现，
    长表比宽表更稳健，便于直接在 Excel/LaTeX 中透视。
    """
    metrics_per_target = result.get("metrics_per_target", {})
    rows: list[dict[str, object]] = []
    for model, splits in metrics_per_target.items():
        for split_name in ("train", "test"):
            per_target = splits.get(split_name, {})
            for target, scores in per_target.items():
                rows.append(
                    {
                        "model": model,
                        "split": split_name,
                        "target": target,
                        "r2": scores.get("r2", float("nan")),
                        "rmse": scores.get("rmse", float("nan")),
                        "mae": scores.get("mae", float("nan")),
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "metrics_per_target.csv", index=False)


def save_cv_metrics(result: dict, output_dir: Path) -> None:
    """导出 ``cv_metrics.csv``：Repeated K-Fold 95% CI。"""
    rows: list[dict[str, object]] = []
    for model, cv in result.get("cv_metrics", {}).items():
        rows.append(
            {
                "model": model,
                "r2_mean": cv.get("r2_mean", float("nan")),
                "r2_ci95_low": cv.get("r2_ci95_low", float("nan")),
                "r2_ci95_high": cv.get("r2_ci95_high", float("nan")),
                "n_splits": cv.get("n_splits"),
                "n_repeats": cv.get("n_repeats"),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "cv_metrics.csv", index=False)


def save_permutation_importance(result: dict, output_dir: Path) -> None:
    """导出 ``permutation_importance.csv`` + 每模型一张条形图。

    Permutation Importance 与 `feature_importances_` 相比的好处是**跨模型统一口径**
    （RF/XGBoost/BP 都能打分），因此适合作为论文中"多模型特征重要性对比"的主要依据；
    树模型内置重要性仍由 ``feature_importance_by_model`` 另行保留。
    """
    pim = result.get("permutation_importance_by_model", {})
    rows: list[dict[str, object]] = []
    for model, per_target in pim.items():
        for target, feature_scores in per_target.items():
            for feature, score in feature_scores.items():
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "feature": feature,
                        "permutation_importance_mean": float(score),
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "permutation_importance.csv", index=False)

    # 每模型画一张横向条形图（按均值大小排序，tensile/shear 并列）
    for model, per_target in pim.items():
        df = pd.DataFrame(per_target).fillna(0.0)
        if df.empty:
            continue
        df = df.reindex(FEATURE_COLUMNS)
        df.index.name = "feature"
        df = df.sort_values(by=df.columns[0], ascending=True)
        ax = df.plot(kind="barh", figsize=(6, 4))
        ax.set_xlabel("Permutation importance (R² drop)")
        ax.set_title(f"Permutation importance — {model}")
        plt.tight_layout()
        plt.savefig(output_dir / f"permutation_importance_{model}.png", dpi=180)
        plt.close()


def save_baseline_metrics(result: dict, output_dir: Path) -> None:
    """导出 ``baseline_metrics.csv``：均值/线性/经验公式基线的测试集指标 + CV 均值。"""
    baseline_metrics = result.get("baseline_metrics", {})
    rows: list[dict[str, object]] = []
    for name, entry in baseline_metrics.items():
        by_target = entry.get("by_target")
        if by_target:
            for target, tentry in by_target.items():
                test_overall = tentry.get("test", {}).get("overall", {})
                cv = tentry.get("cv", {})
                rows.append(
                    {
                        "model": name,
                        "target": target,
                        "test_r2": test_overall.get("r2", float("nan")),
                        "test_rmse": test_overall.get("rmse", float("nan")),
                        "test_mae": test_overall.get("mae", float("nan")),
                        "cv_r2_mean": cv.get("r2_mean", float("nan")),
                        "cv_r2_ci95_low": cv.get("r2_ci95_low", float("nan")),
                        "cv_r2_ci95_high": cv.get("r2_ci95_high", float("nan")),
                    }
                )
            continue
        test_overall = entry.get("test", {}).get("overall", {})
        cv = entry.get("cv", {})
        rows.append(
            {
                "model": name,
                "target": "",
                "test_r2": test_overall.get("r2", float("nan")),
                "test_rmse": test_overall.get("rmse", float("nan")),
                "test_mae": test_overall.get("mae", float("nan")),
                "cv_r2_mean": cv.get("r2_mean", float("nan")),
                "cv_r2_ci95_low": cv.get("r2_ci95_low", float("nan")),
                "cv_r2_ci95_high": cv.get("r2_ci95_high", float("nan")),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "baseline_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_prediction_scatter(result: dict, output_dir: Path) -> None:
    """绘制"实测值-预测值"散点图，用于直观评估模型泛化能力。"""
    pred_source = result.get(
        "predictions_by_model", {result["best_model_name"]: result["predictions"]}
    )
    y_test_by_target = result.get("y_test_by_target")
    active_targets = result.get("active_targets", TARGET_COLUMNS)

    for model, pred_dict in pred_source.items():
        for target in active_targets:
            if target not in pred_dict:
                continue
            if y_test_by_target and target in y_test_by_target:
                y_true = y_test_by_target[target][target].reset_index(drop=True)
            else:
                y_df = result["y_test_df"].reset_index(drop=True)
                if target not in y_df.columns:
                    continue
                y_true = y_df[target]
            pred_series = pd.Series(pred_dict[target])
            plt.figure(figsize=(5, 5))
            plt.scatter(y_true, pred_series, alpha=0.8)
            min_v = min(y_true.min(), pred_series.min())
            max_v = max(y_true.max(), pred_series.max())
            plt.plot([min_v, max_v], [min_v, max_v], "r--")
            plt.xlabel("Measured")
            plt.ylabel("Predicted")
            plt.title(f"{model} - {target}")
            plt.tight_layout()
            plt.savefig(output_dir / f"scatter_{model}_{target}.png", dpi=180)
            plt.close()


# ---------------------------------------------------------------------------
# SHAP  (PR#3 薄壳：委托给 shap_analysis.run_dual_target_shap)
# ---------------------------------------------------------------------------


def run_shap_analysis(result: dict, output_dir: Path) -> dict:
    """薄壳：调用 :func:`uhpc_ml_app.shap_analysis.run_dual_target_shap` 并落盘。

    保留旧函数名与文件产物（``best_model_info.json``），但解释逻辑已换成双目标版本。
    """
    from uhpc_ml_app.shap_analysis import run_dual_target_shap

    output = run_dual_target_shap(result, Path(output_dir))
    with (Path(output_dir) / "best_model_info.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "save_metrics",
    "save_per_target_metrics",
    "save_cv_metrics",
    "save_permutation_importance",
    "save_baseline_metrics",
    "save_prediction_scatter",
    "run_shap_analysis",
]
