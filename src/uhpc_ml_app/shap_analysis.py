"""SHAP 解释与依赖图驱动阈值（PR#3 重写版本）。

相较于原 ``reporting.run_shap_analysis``：

1. **双目标**：拉伸粘结强度 / 抗剪粘结强度分别计算 SHAP，不再只解释第一个子回归器。
2. **按模型类型分派 Explainer**：
   - Tree 家族（RandomForest / XGBRegressor / GradientBoostingRegressor）→ ``TreeExplainer``；
   - Pipeline 包裹的 MLP / 其他通用估计器 → ``KernelExplainer``（用 k-means 背景集）。
3. **依赖图驱动的阈值区间**：以 SHAP 对三维粗糙度 Sa 的取值依赖关系为主，用平滑曲线找"正效应区间"
   并以分位数去尾，替换原 ``_estimate_threshold`` 的朴素 5-桶取最大。
4. **条件化推荐**：可按 ``water_binder_ratio`` / ``curing_age_days`` 分桶给出条件阈值，
   避免报告一个单一的"万能推荐区间"。
5. **显式失败标志**：SHAP 失败时产生 ``explanation_method="correlation_fallback"`` 并记录原因，
   不再默默地把 Pearson 相关系数当成 SHAP 重要性发布。

输出：
- ``shap_importance_by_target.csv``（长表：model / target / feature / mean_abs_shap）
- ``shap_summary_{target}.png``（beeswarm）
- ``shap_dependence_{target}_{feature}.png``（仅对三维粗糙度 Sa 特征）
- ``roughness_recommendations.json``（总体 + 条件化推荐区间）
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

try:  # 绝大多数情况下本地都装了 xgboost；失败容忍是为了与 modeling.py 的回退一致
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS

# 做阈值推荐的"核心关注特征"——粗糙度是论文主变量（三维面粗糙度 Sa）
ROUGHNESS_FEATURES: tuple[str, ...] = ("roughness_sa",)
# 条件化分桶维度
CONDITION_COLUMNS: tuple[str, ...] = ("water_binder_ratio", "curing_age_days")


# ---------------------------------------------------------------------------
# Dispatch: pick an explainer matching the fitted estimator's topology
# ---------------------------------------------------------------------------


@dataclass
class _PerTargetShap:
    """单目标 SHAP 计算结果的轻量封装。"""

    target: str
    explainer_kind: str  # "tree" / "kernel" / "fallback"
    shap_values: np.ndarray  # shape = (n_samples, n_features)
    feature_values: np.ndarray  # shape = (n_samples, n_features)
    feature_names: list[str]


def _is_tree_model(estimator: BaseEstimator) -> bool:
    if isinstance(estimator, (RandomForestRegressor, GradientBoostingRegressor)):
        return True
    if XGBRegressor is not None and isinstance(estimator, XGBRegressor):
        return True
    return False


def _pipeline_predict_fn(pipeline: Pipeline) -> Callable[[np.ndarray], np.ndarray]:
    """包装 Pipeline.predict 供 KernelExplainer 使用。

    对单目标 SHAP，传入的 estimator/pipeline 预测是 (n_samples, n_targets) 矩阵；
    这里只 peel 出特定目标列的闭包在 ``_kernel_explain`` 内生成。
    """

    def _predict(X: np.ndarray) -> np.ndarray:
        return pipeline.predict(X)

    return _predict


def _tree_explain(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    target_idx: int,
) -> np.ndarray:
    """对 Tree-family 估计器返回 (n_samples, n_features) 的 SHAP 值。

    - sklearn ``RandomForestRegressor`` 多输出：TreeExplainer 返回 (n_samples, n_features, n_targets)
      或等价 list，下标到目标即可；
    - ``XGBRegressor``（一个目标）：返回 (n_samples, n_features)；
    - ``GradientBoostingRegressor`` 默认单目标：同上。
    """
    import shap

    explainer = shap.TreeExplainer(estimator)
    raw = explainer.shap_values(x)

    # shap 版本会在不同模型上给出 list / 3D array，这里统一归一成 (n_samples, n_features)
    if isinstance(raw, list):
        return np.asarray(raw[target_idx])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # (n_samples, n_features, n_targets) 或 (n_targets, n_samples, n_features)
        if arr.shape[-1] == len(TARGET_COLUMNS):
            return arr[:, :, target_idx]
        if arr.shape[0] == len(TARGET_COLUMNS):
            return arr[target_idx, :, :]
    return arr


def _kernel_explain(
    pipeline: Pipeline | BaseEstimator,
    x_background: pd.DataFrame,
    x_explain: pd.DataFrame,
    target_idx: int,
    max_background: int = 20,
    max_explain: int = 60,
) -> np.ndarray:
    """KernelExplainer 解释 Pipeline / MLP 等非树模型。

    小样本论文数据（~100 行）下完全够用；大数据时可进一步增大下采样以节流。
    """
    import shap

    background_size = min(max_background, len(x_background))
    try:
        km = KMeans(n_clusters=max(2, min(background_size, 10)), n_init=5, random_state=0)
        km.fit(x_background.to_numpy())
        background = pd.DataFrame(km.cluster_centers_, columns=x_background.columns)
    except Exception:
        background = x_background.sample(n=background_size, random_state=0)

    def _f(X: np.ndarray) -> np.ndarray:
        pred = pipeline.predict(X)
        if pred.ndim == 1:
            return pred
        return pred[:, target_idx]

    sample = x_explain
    if len(sample) > max_explain:
        sample = sample.sample(n=max_explain, random_state=0)

    explainer = shap.KernelExplainer(_f, background)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(sample.to_numpy(), nsamples=100)
    return np.asarray(shap_values)


# ---------------------------------------------------------------------------
# Threshold detection from a (feature_value, shap_value) dependence curve
# ---------------------------------------------------------------------------


def _recommend_threshold_from_dependence(
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    min_samples: int = 6,
    smoothing_window: int | None = None,
) -> dict[str, float | int | str | None]:
    """从 SHAP 依赖曲线中抽取"正效应区间"。

    策略：
    1. 按特征值升序排序 (x, shap)；
    2. 以窗口 ``w = max(min_samples, n // 6)`` 做滑动均值平滑（避免小样本噪声）；
    3. 找平滑 SHAP 值 ≥ 0 的最长连续区间，以两端特征值作为推荐上下界；
    4. 若整段都 <0（该特征全是负效应），则返回 None 并附原因。
    """
    if len(feature_values) < min_samples:
        return {
            "low": None,
            "high": None,
            "reason": "too_few_samples",
            "n_samples": int(len(feature_values)),
        }

    order = np.argsort(feature_values)
    x_sorted = np.asarray(feature_values)[order]
    s_sorted = np.asarray(shap_values)[order]

    n = len(x_sorted)
    if smoothing_window is None:
        smoothing_window = max(min_samples, n // 6)
    smoothing_window = max(2, min(smoothing_window, n))

    # pandas rolling 更稳健，边缘处填 ffill/bfill
    smoothed = (
        pd.Series(s_sorted)
        .rolling(window=smoothing_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    positive_mask = smoothed >= 0
    if not positive_mask.any():
        # 全负效应：退而求其次给出 SHAP 最大点附近的区间
        peak_idx = int(np.argmax(smoothed))
        lo = max(0, peak_idx - smoothing_window // 2)
        hi = min(n - 1, peak_idx + smoothing_window // 2)
        return {
            "low": float(x_sorted[lo]),
            "high": float(x_sorted[hi]),
            "reason": "no_positive_region_report_peak",
            "n_samples": int(n),
            "mean_shap_in_window": float(smoothed[lo : hi + 1].mean()),
        }

    # 寻找最长连续 True 段
    best_start = best_end = -1
    best_len = 0
    cur_start = None
    for i, flag in enumerate(positive_mask):
        if flag and cur_start is None:
            cur_start = i
        elif not flag and cur_start is not None:
            if i - cur_start > best_len:
                best_len = i - cur_start
                best_start, best_end = cur_start, i - 1
            cur_start = None
    if cur_start is not None and n - cur_start > best_len:
        best_start, best_end = cur_start, n - 1

    return {
        "low": float(x_sorted[best_start]),
        "high": float(x_sorted[best_end]),
        "reason": "positive_effect_plateau",
        "n_samples": int(n),
        "mean_shap_in_window": float(smoothed[best_start : best_end + 1].mean()),
    }


def _conditional_recommendations(
    x: pd.DataFrame,
    feature: str,
    shap_vals_feature: np.ndarray,
    condition_col: str,
    n_buckets: int = 3,
) -> list[dict]:
    """按 ``condition_col`` 的分位数桶，在各桶内分别给出 ``feature`` 的推荐区间。"""
    try:
        buckets = pd.qcut(x[condition_col], q=n_buckets, duplicates="drop")
    except ValueError:
        return []
    records: list[dict] = []
    for bucket, sub_index in x.groupby(buckets, observed=True).groups.items():
        mask = x.index.isin(sub_index)
        rec = _recommend_threshold_from_dependence(
            x.loc[mask, feature].to_numpy(),
            shap_vals_feature[mask],
        )
        # pandas.qcut 返回 Interval 对象；mypy 只看到 Hashable 所以这里显式 cast
        interval = pd.Interval(
            left=getattr(bucket, "left", float("nan")),
            right=getattr(bucket, "right", float("nan")),
            closed=getattr(bucket, "closed", "right"),
        )
        records.append(
            {
                "condition_column": condition_col,
                "condition_bucket": f"({interval.left:.3g}, {interval.right:.3g}]",
                "n_samples": int(mask.sum()),
                "feature": feature,
                **rec,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def _unwrap_to_core(model: BaseEstimator) -> BaseEstimator:
    inner = model
    if isinstance(inner, Pipeline):
        inner = inner.steps[-1][1]
    return inner


def _multi_output_estimators(core: BaseEstimator) -> list[BaseEstimator] | None:
    """对于 MultiOutputRegressor 拆出 per-target 子估计器；否则返回 None。"""
    if isinstance(core, MultiOutputRegressor):
        return list(core.estimators_)
    return None


def _explain_for_target(
    best_model: BaseEstimator,
    x_train: pd.DataFrame,
    x_explain: pd.DataFrame,
    target_idx: int,
    target_name: str | None = None,
) -> _PerTargetShap:
    """按 best_model 的实际类型选择解释器，返回 (n_samples, n_features) SHAP 值。"""
    core = _unwrap_to_core(best_model)
    resolved_target = target_name or TARGET_COLUMNS[target_idx]
    multi = _multi_output_estimators(core)

    # case 1: MultiOutputRegressor 包 tree 子估计器（典型 XGBoost 路径）
    if multi is not None and all(_is_tree_model(est) for est in multi):
        try:
            shap_arr = _tree_explain(multi[target_idx], x_explain, target_idx=0)
            return _PerTargetShap(
                target=resolved_target,
                explainer_kind="tree",
                shap_values=shap_arr,
                feature_values=x_explain.to_numpy(),
                feature_names=list(x_explain.columns),
            )
        except Exception as exc:
            # 例如 shap 0.49 + xgboost 3.x 的 base_score 解析不兼容 → 回退到 KernelExplainer
            warnings.warn(
                f"TreeExplainer failed for {target_name}, falling back to KernelExplainer: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    # case 2: 原生多输出 tree（RandomForest 默认）
    elif multi is None and _is_tree_model(core):
        try:
            shap_arr = _tree_explain(core, x_explain, target_idx=target_idx)
            return _PerTargetShap(
                target=resolved_target,
                explainer_kind="tree",
                shap_values=shap_arr,
                feature_values=x_explain.to_numpy(),
                feature_names=list(x_explain.columns),
            )
        except Exception as exc:
            warnings.warn(
                f"TreeExplainer failed for {target_name}, falling back to KernelExplainer: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    # case 3: Pipeline 包 MLP 或 tree fallback——KernelExplainer
    shap_arr = _kernel_explain(best_model, x_train, x_explain, target_idx=target_idx)
    # KernelExplainer 可能在 x_explain 子采样，所以要同步特征值
    if shap_arr.shape[0] != x_explain.shape[0]:
        # 重新取样一遍使二者一致（确定性种子）
        if len(x_explain) > shap_arr.shape[0]:
            x_used = x_explain.sample(n=shap_arr.shape[0], random_state=0)
        else:
            x_used = x_explain
    else:
        x_used = x_explain
    return _PerTargetShap(
        target=resolved_target,
        explainer_kind="kernel",
        shap_values=shap_arr,
        feature_values=x_used.to_numpy(),
        feature_names=list(x_used.columns),
    )


def _mean_abs_importance(per_target: _PerTargetShap) -> dict[str, float]:
    mean_abs = np.abs(per_target.shap_values).mean(axis=0)
    return dict(zip(per_target.feature_names, map(float, mean_abs)))


def run_dual_target_shap(result: dict, output_dir: Path) -> dict:
    """主入口：对 ``result['best_model']`` 同时解释拉伸与抗剪。

    返回写入 ``best_model_info.json`` 的结构化字典（包括 per-target SHAP 重要性
    与 per-target 粗糙度推荐区间）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_by_target: dict[str, BaseEstimator] = result.get(
        "best_model_by_target",
        {TARGET_COLUMNS[0]: result["best_model"]},
    )
    target_runs: dict = result.get("target_runs", {})
    active_targets = result.get("active_targets", list(best_model_by_target.keys()))

    shap_by_target: dict[str, _PerTargetShap] = {}
    explanation_methods: dict[str, str] = {}
    failure_reasons: dict[str, str] = {}

    try:
        import shap  # noqa: F401 imported for the explainers above
    except Exception as exc:  # shap 未安装时直接走 fallback
        return _correlation_fallback(result, output_dir, reason=f"shap_import_failed:{exc}")

    for target in active_targets:
        if target not in best_model_by_target:
            continue
        best_model = best_model_by_target[target]
        run = target_runs.get(target, result)
        x_train_t: pd.DataFrame = run["x_train"]
        x_eval_t: pd.DataFrame = run.get("x_test", x_train_t)
        try:
            per_target = _explain_for_target(
                best_model, x_train_t, x_eval_t, target_idx=0, target_name=target
            )
            shap_by_target[target] = per_target
            explanation_methods[target] = per_target.explainer_kind
        except Exception as exc:  # 单目标失败不拖累另一目标
            failure_reasons[target] = f"{type(exc).__name__}: {exc}"
            explanation_methods[target] = "failed"

    if not shap_by_target:
        return _correlation_fallback(
            result,
            output_dir,
            reason=f"all_targets_failed:{failure_reasons}",
        )

    # 1. 长表 importance CSV
    rows: list[dict] = []
    for target, pt in shap_by_target.items():
        for feature, score in _mean_abs_importance(pt).items():
            rows.append(
                {
                    "model": result["best_model_name"],
                    "target": target,
                    "feature": feature,
                    "mean_abs_shap": float(score),
                    "explainer_kind": pt.explainer_kind,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "shap_importance_by_target.csv", index=False)

    # 2. summary plot & dependence plot
    for target, pt in shap_by_target.items():
        try:
            import shap as _shap

            _shap.summary_plot(
                pt.shap_values,
                pd.DataFrame(pt.feature_values, columns=pt.feature_names),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_summary_{target}.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        for feat in ROUGHNESS_FEATURES:
            if feat not in pt.feature_names:
                continue
            j = pt.feature_names.index(feat)
            plt.figure(figsize=(5.0, 4.0))
            plt.scatter(pt.feature_values[:, j], pt.shap_values[:, j], alpha=0.7)
            plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            plt.xlabel(feat)
            plt.ylabel(f"SHAP value for {target}")
            plt.title(f"SHAP dependence — {target} vs {feat}")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"shap_dependence_{target}_{feat}.png", dpi=180, bbox_inches="tight"
            )
            plt.close()

    # 3. 依赖曲线驱动的阈值推荐（总体 + 条件化）
    roughness_recommendations: dict[str, dict] = {}
    conditional_rows: list[dict] = []
    for target, pt in shap_by_target.items():
        roughness_recommendations[target] = {}
        x_used = pd.DataFrame(pt.feature_values, columns=pt.feature_names)
        for feat in ROUGHNESS_FEATURES:
            if feat not in pt.feature_names:
                continue
            j = pt.feature_names.index(feat)
            overall = _recommend_threshold_from_dependence(
                pt.feature_values[:, j], pt.shap_values[:, j]
            )
            roughness_recommendations[target][feat] = {
                "overall": overall,
                "conditional": {},
            }
            for cond_col in CONDITION_COLUMNS:
                if cond_col not in pt.feature_names:
                    continue
                cond_rows = _conditional_recommendations(
                    x_used, feat, pt.shap_values[:, j], cond_col
                )
                roughness_recommendations[target][feat]["conditional"][cond_col] = cond_rows
                for row in cond_rows:
                    conditional_rows.append(
                        {"model": result["best_model_name"], "target": target, **row}
                    )
    if conditional_rows:
        pd.DataFrame(conditional_rows).to_csv(
            output_dir / "roughness_conditional_recommendations.csv", index=False
        )
    with (output_dir / "roughness_recommendations.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": result["best_model_name"],
                "recommendations": roughness_recommendations,
                "explanation_method": explanation_methods,
                "failed_targets": failure_reasons,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 4. 顶层摘要（best_model_info.json 的补丁）
    shap_importance_flat: dict[str, float] = {}
    for target, pt in shap_by_target.items():
        for feat, score in _mean_abs_importance(pt).items():
            shap_importance_flat[f"{target}__{feat}"] = score

    primary_recommendation: dict[str, float | None] = {"low": None, "high": None}
    if TARGET_COLUMNS[0] in roughness_recommendations:
        primary_recommendation = {
            k: roughness_recommendations[TARGET_COLUMNS[0]]
            .get("roughness_sa", {})
            .get("overall", {})
            .get(k)
            for k in ("low", "high")
        }

    return {
        "best_model": result["best_model_name"],
        "model_selection_metric": result.get("model_selection_metric", "cv_r2_mean"),
        "explanation_method": explanation_methods,
        "failed_targets": failure_reasons,
        "shap_importance_by_target": {
            target: _mean_abs_importance(pt) for target, pt in shap_by_target.items()
        },
        "shap_importance": shap_importance_flat,  # 向下兼容旧 key
        "roughness_recommendations": roughness_recommendations,
        "roughness_sa_recommendation": {  # 向下兼容旧 key（只含 tensile/Sa）
            "recommended_low": primary_recommendation["low"],
            "recommended_high": primary_recommendation["high"],
        },
        "cv_metrics": result.get("cv_metrics", {}).get(result["best_model_name"], {}),
        "metrics_per_target": result.get("metrics_per_target", {}).get(
            result["best_model_name"], {}
        ),
    }


def _correlation_fallback(result: dict, output_dir: Path, reason: str) -> dict:
    """SHAP 彻底不可用时的相关系数兜底——显式写明 ``explanation_method``。"""
    target_runs = result.get("target_runs", {})
    active_targets = result.get("active_targets", TARGET_COLUMNS)
    importance_by_target = {}
    flat: dict[str, float] = {}

    for target in active_targets:
        run = target_runs.get(target, result)
        x_train = run["x_train"]
        y_train = run["y_train"]
        if target not in y_train.columns:
            continue
        corr = pd.concat([x_train, y_train[[target]]], axis=1).corr(numeric_only=True)
        series = corr[target].reindex(FEATURE_COLUMNS).fillna(0.0).abs()
        importance_by_target[target] = {k: float(v) for k, v in series.to_dict().items()}
        for feat, score in importance_by_target[target].items():
            flat[f"{target}__{feat}"] = score

    best_name_by_target = result.get("best_model_name_by_target", {})
    default_model = result["best_model_name"]
    pd.DataFrame(
        [
            {
                "model": best_name_by_target.get(target, default_model),
                "target": target,
                "feature": feature,
                "mean_abs_shap": score,
                "explainer_kind": "correlation_fallback",
            }
            for target, feats in importance_by_target.items()
            for feature, score in feats.items()
        ]
    ).to_csv(output_dir / "shap_importance_by_target.csv", index=False)
    return {
        "best_model": result["best_model_name"],
        "best_model_name_by_target": best_name_by_target,
        "model_selection_metric": result.get("model_selection_metric", "cv_r2_mean"),
        "explanation_method": dict.fromkeys(active_targets, "correlation_fallback"),
        "failed_targets": {"_global": reason},
        "shap_importance_by_target": importance_by_target,
        "shap_importance": flat,
        "roughness_recommendations": {},
        "roughness_sa_recommendation": {"recommended_low": None, "recommended_high": None},
        "cv_metrics": result.get("cv_metrics", {}).get(result["best_model_name"], {}),
        "metrics_per_target": result.get("metrics_per_target", {}).get(
            result["best_model_name"], {}
        ),
    }


__all__ = [
    "run_dual_target_shap",
    "ROUGHNESS_FEATURES",
    "CONDITION_COLUMNS",
]
