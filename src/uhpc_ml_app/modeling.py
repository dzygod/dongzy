"""\u5efa\u6a21\u4e3b\u6a21\u5757\uff1a\u8bad\u7ec3\u3001\u8c03\u4f18\u3001\u8bc4\u4f30\u3001\u7279\u5f81\u91cd\u8981\u6027\u7edf\u4e00\u3002

PR#2 \u5bf9\u6bd4\u539f\u5b9e\u73b0\u7684\u5173\u952e\u6539\u52a8\uff1a

1. BP \u795e\u7ecf\u7f51\u7edc \u2192 ``Pipeline(StandardScaler + MLPRegressor)``\uff0c\u4e0d\u518d\u56e0\u672a\u6807\u51c6\u5316
   \u800c\u5728 RF/XGBoost \u9762\u524d\u5403\u4e8f\u3002
2. RandomForestRegressor \u4e0d\u518d\u88ab ``MultiOutputRegressor`` \u5305\u88f9\uff0c\u76f4\u63a5\u4f7f\u7528
   sklearn \u539f\u751f\u591a\u8f93\u51fa\u652f\u6301\uff0c\u51cf\u5c11\u8bad\u7ec3\u6210\u672c\u5e76\u4fdd\u7559\u76ee\u6807\u95f4\u7684\u8026\u5408\u4fe1\u606f\u3002
3. \u7f51\u683c\u641c\u7d22\u7a7a\u95f4\u6269\u5927\uff08``n_estimators`` / ``max_features`` / ``reg_alpha`` / ``reg_lambda`` \u7b49\uff09\uff0c
   \u907f\u514d\u539f\u7248\u56e0\u7f51\u683c\u8fc7\u5c0f\u9020\u6210\u7684\u201c\u4f2a\u8c03\u53c2\u201d\u3002
4. \u6307\u6807\u62c6\u5206\u4e3a **per-target + overall**\uff0c\u907f\u514d\u62c9\u4f38/\u6297\u526a\u4e24\u4e2a\u76ee\u6807\u7684\u8868\u73b0
   \u88ab uniform_average \u63a9\u76d6\u3002
5. \u65b0\u589e **Repeated K-Fold CV**\uff08\u9ed8\u8ba4 5\u00d710\uff09\u5728\u5168\u91cf\u6570\u636e\u4e0a\u4e3a\u6700\u4f18\u6a21\u578b
   \u7ed9\u51fa\u83b7\u5f97\u5747\u503c\u00b195% CI \u7684\u6cdb\u5316\u4f30\u8ba1\uff0c\u907f\u514d\u5355\u6b21 20 \u884c\u6d4b\u8bd5\u96c6\u9009\u6a21\u578b\u7684
   \u9ad8\u65b9\u5dee\u95ee\u9898\u3002
6. **\u6a21\u578b\u9009\u4f18\u6539\u4e3a\u6309 CV-R\u00b2 \u5747\u503c**\uff0c\u800c\u975e\u5355\u6b21\u6d4b\u8bd5\u96c6 R\u00b2\u3002
7. \u7edf\u4e00\u5404\u6a21\u578b\u7684\u7279\u5f81\u91cd\u8981\u6027\u4e3a **Permutation Importance**\uff0c\u6811\u6a21\u578b\u7684
   ``feature_importances_`` \u5141\u8bb8\u540c\u65f6\u4fdd\u7559\u4f5c\u4e3a\u53c2\u8003\u3002
8. \u53ef\u9009\u5730\u5e76\u5165\u57fa\u7ebf\u6a21\u578b\uff08\u5747\u503c / \u7ebf\u6027 / \u7ecf\u9a8c\u516c\u5f0f\uff09\u4f5c\u4e3a\u5bf9\u6bd4\u4e0b\u754c\u3002

\u8f93\u51fa\u7684 ``result`` \u5b57\u5178\u4fdd\u6301\u5bf9 reporting.py / streamlit_app.py \u7684**\u5411\u4e0b\u517c\u5bb9**
\uff08``metrics / best_model / x_train / y_train / y_test_df / predictions_by_model`` \u7b49\u952e\u540d\u5747\u5b58\u5728\uff09\uff0c
\u540c\u65f6\u65b0\u589e\u4e86\uff1a``metrics_per_target``\u3001``cv_metrics``\u3001``permutation_importance_by_model``\u3001
``baseline_metrics``\u3002SHAP \u7528\u5230\u7684 ``best_model`` \u4ecd\u4e3a\u6700\u4f18 pipeline/estimator\uff0c
PR#3 \u4f1a\u57fa\u4e8e\u8be5\u5bf9\u8c61\u505a\u53cc\u76ee\u6807\u89e3\u91ca\u3002
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment,misc]

from uhpc_ml_app.baselines import build_baselines
from uhpc_ml_app.config import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    SPARSE_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TrainConfig,
)
from uhpc_ml_app.data_augmentation import augment_regression

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _evaluate_overall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """\u4ee5 uniform_average \u7684\u5f62\u5f0f\u8fd4\u56de\u6574\u4f53\u56de\u5f52\u6307\u6807\uff08\u4e0e\u539f\u7248\u5411\u4e0b\u517c\u5bb9\uff09\u3002"""
    return {
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _evaluate_per_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """\u6309\u76ee\u6807\u5206\u522b\u8ba1\u7b97 R\u00b2 / RMSE / MAE\u3002

    \u539f\u7248\u5c06\u62c9\u4f38\u4e0e\u6297\u526a\u4e24\u4e2a\u76ee\u6807\u5e73\u5747\u5728\u4e00\u8d77\u6c47\u62a5\uff0c\u7531\u4e8e\u4e24\u76ee\u6807\u91cf\u7eb2\u76f8\u5dee
    ~4 \u500d\uff0cRMSE/MAE \u4f1a\u88ab\u6297\u526a\u4e3b\u5bfc\uff0c\u5904\u7406\u4e0a\u4e0d\u4e25\u8c28\u3002\u672c\u51fd\u6570\u4e3a per-target \u62c6\u5206\uff0c
    \u6446\u5230\u8bba\u6587\u8868\u683c\u65f6\u4e5f\u66f4\u5bb9\u6613\u89e3\u8bfb\u3002
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    out: dict[str, dict[str, float]] = {}
    for idx, target in enumerate(target_names):
        yt = y_true_arr[:, idx]
        yp = y_pred_arr[:, idx]
        out[target] = {
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
        }
    return out


def _evaluate_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
) -> dict[str, object]:
    """\u7edf\u4e00\u8fd4\u56de overall + per_target \u4e24\u5c42\u6307\u6807\uff0c\u65b9\u4fbf reporting \u8f93\u51fa\u3002"""
    return {
        "overall": _evaluate_overall(y_true, y_pred),
        "per_target": _evaluate_per_target(y_true, y_pred, target_names),
    }


def _ci95_from_scores(scores: np.ndarray) -> tuple[float, float, float]:
    """\u7531\u6837\u672c\u7ecf\u9a8c\u6807\u51c6\u5dee\u7ed9\u51fa\u5747\u503c + \u8fd1\u4f3c 95% CI\u3002"""
    mean = float(np.mean(scores))
    if len(scores) <= 1:
        return mean, mean, mean
    sem = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _repeated_kfold_cv_scores(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg: TrainConfig,
    random_state: int,
) -> dict[str, float | int]:
    """Repeated K-Fold \u5168\u6570\u636e\u4ea4\u53c9\u9a8c\u8bc1\uff0c\u8fd4\u56de R\u00b2 \u5747\u503c\u00b195% CI \u53ca\u8bbe\u7f6e\u3002

    \u4f7f\u7528 sklearn \u7684 ``cross_val_score(scoring="r2")``\u3002\u8bc4\u4f30\u65f6 ``estimator`` \u4f1a\u88ab\u5185\u90e8
    clone + fit\uff0c\u4e0d\u4f1a\u6c61\u67d3\u5916\u90e8\u7684\u5df2\u62df\u5408\u6a21\u578b\u3002
    """
    rkf = RepeatedKFold(
        n_splits=cfg.repeated_cv_n_splits,
        n_repeats=cfg.repeated_cv_n_repeats,
        random_state=random_state,
    )
    try:
        scores = cross_val_score(
            estimator,
            X,
            y,
            cv=rkf,
            scoring="r2",
            n_jobs=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "r2_mean": float("nan"),
            "r2_ci95_low": float("nan"),
            "r2_ci95_high": float("nan"),
            "n_splits": float(cfg.repeated_cv_n_splits),
            "n_repeats": float(cfg.repeated_cv_n_repeats),
            "error": str(exc),  # type: ignore[dict-item]
        }
    mean, low, high = _ci95_from_scores(scores)
    return {
        "r2_mean": mean,
        "r2_ci95_low": low,
        "r2_ci95_high": high,
        "n_splits": cfg.repeated_cv_n_splits,
        "n_repeats": cfg.repeated_cv_n_repeats,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_models(random_state: int, cfg: TrainConfig) -> dict[str, tuple[BaseEstimator, dict]]:
    """\u6784\u9020\u4e09\u7c7b\u5019\u9009\u6a21\u578b\u53ca\u5176\u7f51\u683c\u641c\u7d22\u53c2\u6570\u7a7a\u95f4\u3002

    \u7ea6\u5b9a\uff1a
    - RF\uff1asklearn \u539f\u751f\u591a\u8f93\u51fa\uff0c\u4e0d\u505a MultiOutputRegressor \u5305\u88f9\u3002
    - XGBoost\uff1a\u4ecd\u4ee5 ``MultiOutputRegressor(XGBRegressor)`` \u63a5\u5165\uff0c\u4fdd\u6301\u7a33\u5b9a
      \u517c\u5bb9\u6027\uff1bxgboost \u4e0d\u53ef\u7528\u65f6\u56de\u9000 ``GradientBoostingRegressor``\u3002
    - BP\uff1a``Pipeline([scaler, mlp])``\uff0c\u4ee5\u4fdd\u8bc1\u5165\u7f51\u524d\u505a\u6807\u51c6\u5316\uff0c\u4e0e\u6811\u6a21\u578b\u516c\u5e73\u5bf9\u6bd4\u3002
    """
    fast = cfg.grid_mode == "fast"
    rf_estimator = RandomForestRegressor(random_state=random_state, n_jobs=1)
    rf_grid = (
        {
            "n_estimators": [200],
            "max_depth": [None, 6],
            "min_samples_split": [2],
        }
        if fast
        else {
            "n_estimators": [200, 500],
            "max_depth": [None, 6, 12],
            "min_samples_split": [2, 4],
        }
    )

    if XGBRegressor is not None:
        xgb_estimator = XGBRegressor(
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=1,
            verbosity=0,
        )
        xgb_grid = (
            {
                "n_estimators": [200],
                "max_depth": [3, 5],
                "learning_rate": [0.1],
                "subsample": [1.0],
            }
            if fast
            else {
                "n_estimators": [200, 500],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
            }
        )
    else:
        xgb_estimator = GradientBoostingRegressor(random_state=random_state)
        xgb_grid = (
            {
                "n_estimators": [200],
                "max_depth": [2, 3],
                "learning_rate": [0.1],
                "subsample": [1.0],
            }
            if fast
            else {
                "n_estimators": [200, 500],
                "max_depth": [2, 3],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
            }
        )

    bp_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    random_state=random_state,
                    max_iter=cfg.mlp_max_iter,
                    early_stopping=False,
                ),
            ),
        ]
    )
    bp_grid = (
        {
            "mlp__hidden_layer_sizes": [(32,)],
            "mlp__activation": ["relu"],
            "mlp__alpha": [1e-4],
        }
        if fast
        else {
            "mlp__hidden_layer_sizes": [(64,), (64, 32)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [1e-4, 1e-3],
        }
    )

    return {
        "random_forest": (rf_estimator, rf_grid),
        "xgboost": (xgb_estimator, xgb_grid),
        "bp_neural_network": (bp_pipeline, bp_grid),
    }


# ---------------------------------------------------------------------------
# Feature importance (unified + native)
# ---------------------------------------------------------------------------


def _tree_native_importance_single(
    model_name: str,
    fitted: BaseEstimator,
    feature_names: list[str],
    target: str,
) -> dict[str, dict[str, float]]:
    core: BaseEstimator = fitted
    if isinstance(fitted, Pipeline):
        core = fitted.steps[-1][1]
    scores = getattr(core, "feature_importances_", None)
    if scores is None:
        normalized = dict.fromkeys(feature_names, 0.0)
    else:
        normalized = {f: float(s) for f, s in zip(feature_names, scores)}
    return {target: normalized}


def _permutation_importance_single(
    fitted: BaseEstimator,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    feature_names: list[str],
    cfg: TrainConfig,
    random_state: int,
) -> dict[str, float]:
    try:
        pi = permutation_importance(
            fitted,
            X_eval,
            y_eval,
            scoring="r2",
            n_repeats=cfg.permutation_importance_n_repeats,
            random_state=random_state,
            n_jobs=1,
        )
        scores = pi.importances_mean
    except Exception:  # pragma: no cover
        scores = np.zeros(len(feature_names))
    return {f: float(s) for f, s in zip(feature_names, scores)}


def _evaluate_single(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _evaluate_baseline_single(
    name: str,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: TrainConfig,
    random_state: int,
) -> dict[str, object]:
    y_tr = y_train.to_frame()
    y_te = y_test.to_frame()
    fitted = clone(estimator).fit(x_train, y_tr)
    train_pred = np.asarray(fitted.predict(x_train)).ravel()
    test_pred = np.asarray(fitted.predict(x_test)).ravel()
    cv_metrics = _repeated_kfold_cv_scores(
        clone(estimator),
        pd.concat([x_train, x_test], axis=0).reset_index(drop=True),
        pd.concat([y_tr, y_te], axis=0).reset_index(drop=True),
        cfg,
        random_state,
    )
    return {
        "name": name,
        "train": {"overall": _evaluate_single(y_train, train_pred)},
        "test": {"overall": _evaluate_single(y_test, test_pred)},
        "cv": cv_metrics,
    }


def _train_one_target(
    df: pd.DataFrame,
    target: str,
    cfg: TrainConfig,
    random_state: int,
) -> dict:
    """对单个输出目标（拉伸或抗剪）独立训练三模型。"""
    x = df[FEATURE_COLUMNS]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=random_state
    )

    # 可选数据增强：仅作用于训练集，测试集保持真实数据（默认关闭）
    if cfg.augment:
        x_train, y_train = augment_regression(
            x_train,
            y_train,
            n_synthetic=cfg.augment_n_synthetic,
            numeric_columns=NUMERIC_FEATURE_COLUMNS,
            sparse_columns=SPARSE_FEATURE_COLUMNS,
            noise_scale=cfg.augment_noise_scale,
            random_state=random_state,
        )

    models = _build_models(random_state, cfg)
    metrics: dict[str, dict] = {}
    cv_metrics: dict[str, dict] = {}
    fitted: dict[str, BaseEstimator] = {}
    predictions: dict[str, dict[str, list[float]]] = {}
    native_importance: dict[str, dict[str, dict[str, float]]] = {}
    permutation_importance_by_model: dict[str, dict[str, dict[str, float]]] = {}

    inner_cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=random_state)

    for name, (estimator, grid) in models.items():
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=inner_cv,
            scoring="r2",
            n_jobs=1,
            refit=True,
        )
        gs.fit(x_train, y_train)
        best = gs.best_estimator_
        fitted[name] = best

        train_pred = np.asarray(best.predict(x_train)).ravel()
        test_pred = np.asarray(best.predict(x_test)).ravel()

        metrics[name] = {
            "train": _evaluate_single(y_train, train_pred),
            "test": _evaluate_single(y_test, test_pred),
            "best_params": gs.best_params_,
        }
        cv_metrics[name] = _repeated_kfold_cv_scores(
            clone(estimator).set_params(**gs.best_params_),
            x,
            y.to_frame(),
            cfg,
            random_state,
        )
        native_importance[name] = _tree_native_importance_single(
            name, best, FEATURE_COLUMNS, target
        )
        permutation_importance_by_model[name] = {
            target: _permutation_importance_single(
                best, x_test, y_test, FEATURE_COLUMNS, cfg, random_state
            )
        }
        predictions[name] = {target: test_pred.astype(float).tolist()}

    def _model_selection_score(name: str) -> float:
        if cfg.model_selection_metric == "cv_r2_mean":
            val = cv_metrics.get(name, {}).get("r2_mean", float("nan"))
            return float(val) if val == val else metrics[name]["test"]["r2"]
        return metrics[name]["test"]["r2"]

    best_model_name = max(fitted.keys(), key=_model_selection_score)

    baseline_metrics: dict[str, dict[str, object]] = {}
    if cfg.include_baselines:
        for bname, bmodel in build_baselines().items():
            baseline_metrics[bname] = _evaluate_baseline_single(
                bname, bmodel, x_train, y_train, x_test, y_test, cfg, random_state
            )

    return {
        "target": target,
        "metrics": metrics,
        "metrics_per_target": {
            name: {
                "train": {target: metrics[name]["train"]},
                "test": {target: metrics[name]["test"]},
            }
            for name in fitted
        },
        "cv_metrics": cv_metrics,
        "fitted": fitted,
        "predictions_by_model": predictions,
        "best_model_name": best_model_name,
        "best_model": fitted[best_model_name],
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train.to_frame(name=target),
        "y_test_df": y_test.to_frame(name=target),
        "feature_importance_by_model": native_importance,
        "permutation_importance_by_model": permutation_importance_by_model,
        "baseline_metrics": baseline_metrics,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_compare_models(
    df: pd.DataFrame,
    random_state: int = 42,
    config: TrainConfig | None = None,
) -> dict:
    """按输出目标独立训练：拉伸与抗剪分别建模，允许样本只含其一。"""
    cfg = config or TrainConfig()
    target_runs: dict[str, dict] = {}

    for target in TARGET_COLUMNS:
        sub = df.loc[df[target].notna()].copy()
        if len(sub) < 8:
            continue
        target_runs[target] = _train_one_target(sub, target, cfg, random_state)

    if not target_runs:
        raise ValueError("拉伸/抗剪标签均不足 8 条，无法训练。")

    primary_target = (
        TARGET_COLUMNS[0] if TARGET_COLUMNS[0] in target_runs else next(iter(target_runs))
    )
    model_names = list(next(iter(target_runs.values()))["metrics"].keys())
    metrics: dict[str, dict] = {}
    metrics_per_target: dict[str, dict] = {}
    cv_metrics: dict[str, dict] = {}
    predictions_by_model: dict[str, dict[str, list[float]]] = {m: {} for m in model_names}
    native_importance: dict[str, dict[str, dict[str, float]]] = {}
    permutation_importance_by_model: dict[str, dict[str, dict[str, float]]] = {
        m: {} for m in model_names
    }
    baseline_metrics: dict[str, dict[str, object]] = {}

    for model in model_names:
        train_r2s: list[float] = []
        test_r2s: list[float] = []
        cv_r2s: list[float] = []
        metrics_per_target[model] = {"train": {}, "test": {}}
        native_importance[model] = {}

        for target, run in target_runs.items():
            m = run["metrics"][model]
            metrics_per_target[model]["train"][target] = m["train"]
            metrics_per_target[model]["test"][target] = m["test"]
            train_r2s.append(m["train"]["r2"])
            test_r2s.append(m["test"]["r2"])
            cv = run["cv_metrics"][model]
            cv_r2s.append(float(cv.get("r2_mean", float("nan"))))
            predictions_by_model[model][target] = run["predictions_by_model"][model][target]
            native_importance[model][target] = run["feature_importance_by_model"][model][target]
            permutation_importance_by_model[model][target] = run["permutation_importance_by_model"][
                model
            ][target]

        metrics[model] = {
            "train": {
                "r2": float(np.mean(train_r2s)),
                "rmse": float("nan"),
                "mae": float("nan"),
            },
            "test": {
                "r2": float(np.mean(test_r2s)),
                "rmse": float("nan"),
                "mae": float("nan"),
            },
            "best_params": target_runs[next(iter(target_runs))]["metrics"][model]["best_params"],
        }
        _cv = [run["cv_metrics"][model] for run in target_runs.values()]
        cv_metrics[model] = {
            "r2_mean": float(np.mean([c["r2_mean"] for c in _cv])),
            "r2_ci95_low": float(np.mean([c["r2_ci95_low"] for c in _cv])),
            "r2_ci95_high": float(np.mean([c["r2_ci95_high"] for c in _cv])),
            "n_splits": cfg.repeated_cv_n_splits,
            "n_repeats": cfg.repeated_cv_n_repeats,
        }

    best_model_name_by_target = {t: run["best_model_name"] for t, run in target_runs.items()}
    best_model_by_target = {t: run["best_model"] for t, run in target_runs.items()}

    best_model_name = best_model_name_by_target[primary_target]
    best_model = best_model_by_target[primary_target]
    primary_run = target_runs[primary_target]

    if cfg.include_baselines:
        for bname in next(iter(target_runs.values()))["baseline_metrics"]:
            baseline_metrics[bname] = {
                "name": bname,
                "by_target": {t: run["baseline_metrics"][bname] for t, run in target_runs.items()},
            }

    feature_importance = native_importance[best_model_name]
    predictions_best = predictions_by_model[best_model_name]

    return {
        "config": asdict(cfg),
        "active_targets": list(target_runs.keys()),
        "target_runs": target_runs,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "feature_importance_by_model": native_importance,
        "predictions": predictions_best,
        "predictions_by_model": predictions_by_model,
        "y_test_by_target": {t: run["y_test_df"] for t, run in target_runs.items()},
        "y_test": primary_run["y_test_df"].to_dict(orient="list"),
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_model_name_by_target": best_model_name_by_target,
        "best_model_by_target": best_model_by_target,
        "x_train": primary_run["x_train"],
        "x_test": primary_run["x_test"],
        "y_train": primary_run["y_train"],
        "y_test_df": primary_run["y_test_df"],
        "metrics_per_target": metrics_per_target,
        "cv_metrics": cv_metrics,
        "permutation_importance_by_model": permutation_importance_by_model,
        "baseline_metrics": baseline_metrics,
        "model_selection_metric": cfg.model_selection_metric,
    }


__all__ = [
    "train_and_compare_models",
]
