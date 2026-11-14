"""小论文 1 早期流程演示：XGBoost + 对照模型，5 折 × 30 重复 CV。

当前稿件的 467 条主分析样本及 484 条扩展分析样本和报告指标请使用 paper1_group_validation_v4.py。
本脚本使用较早的数据清洗口径，输出不能对应当前稿件：

- 数据：500 行原始记录经库内 ``load_and_clean_data`` 清洗，样本数
  与当前稿件的筛选口径不同。真实数据不含于本仓库，
  需按 ``docs/data_protocol.md`` 自行整理，或向通讯作者合理请求（见 README §11）。
  仓库自带的 ``data/raw/literature_sample.csv`` 为合成示例，仅用于验证脚本可运行，
  **禁止将基于它的输出用于论文结论**。
- 特征：``src/uhpc_ml_app/config.py`` 的 ``FEATURE_COLUMNS``（4 连续数值特征 +
  ``roughness_mm`` 稀疏宏观纹理 + ``interface_prep`` / ``specimen_type`` 的 one-hot）。
- 模型：XGBoost（主）+ RandomForest / BP-MLP / 均值·线性基线（对照）。
  XGBoost 固定超参：max_depth=5, learning_rate=0.02, n_estimators=300,
  subsample=0.8, colsample_bytree=0.8。
- 评估：5 折 CV × 30 次重复（random_state=42）。
  - pooled OOF R²：把每重复内 5 折袋外(out-of-fold)预测拼成一条算一次 R²，
    再对 30 次重复取均值 ± 标准差。
  - 逐折平均 R²（fold_mean_r2）：数值略低，属正常，二者都诚实。
  - RMSE / MAE：在"全部折+全部重复"拼接的 pooled OOF 预测上计算。

旧流程运行示例（真实数据）：:

    python scripts/reproduce_paper1.py --csv <path>/uhpc_nc_bond_literature_data_v9.csv --out outputs/run-legacy-demo

跑法（合成示例，仅验证流程）：:

    python scripts/reproduce_paper1.py
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 无论是否 pip install，都能定位到仓库根目录下的 src 包
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from uhpc_ml_app.config import FEATURE_COLUMNS
from uhpc_ml_app.data_pipeline import load_and_clean_data

DATA_CSV = str(_REPO_ROOT / "data" / "raw" / "literature_sample.csv")
OUT_DIR = _REPO_ROOT / "outputs" / "run-paper1"

SEED = 42
N_SPLITS = 5
N_REPEATS = 30

# 论文固化的 XGBoost 超参
XGB_PARAMS = dict(
    max_depth=5,
    learning_rate=0.02,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
)

# 对照模型固定配置（非网格搜索；XGBoost 为论文主模型）
RF_PARAMS = dict(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=SEED, n_jobs=1)
MLP_PARAMS = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=2000, random_state=SEED)

TARGETS = ["shear_bond_strength", "tensile_bond_strength"]  # 抗剪为主，抗拉为辅

FEATURE_LABELS = {
    "water_binder_ratio": "w/b ratio",
    "fiber_volume_ratio": "fiber volume",
    "nc_strength_grade": "NC strength",
    "curing_age_days": "curing age",
    "roughness_mm": "roughness (mm)",
    "interface_prep__C": "prep: untreated",
    "interface_prep__SB": "prep: sandblast",
    "interface_prep__HW": "prep: hydrodemolition",
    "interface_prep__CH": "prep: chipping",
    "interface_prep__M": "prep: milling",
    "specimen_type__pull_off": "type: pull-off",
    "specimen_type__slant_shear": "type: slant shear",
    "specimen_type__splitting": "type: splitting",
    "specimen_type__bi_shear": "type: bi-shear",
    "specimen_type__direct_shear": "type: direct shear",
    "specimen_type__direct_tension": "type: direct tension",
    "specimen_type__bending": "type: bending",
    "specimen_type__mixed": "type: mixed",
}


def model_factory(name: str):
    """按名字返回"全新未拟合"估计器工厂（每次 CV 折重新拟合）。"""
    if name == "xgboost":
        return lambda: XGBRegressor(**XGB_PARAMS)
    if name == "random_forest":
        return lambda: RandomForestRegressor(**RF_PARAMS)
    if name == "bp_mlp":
        return lambda: Pipeline(
            [("scaler", StandardScaler()), ("mlp", MLPRegressor(**MLP_PARAMS))]
        )
    if name == "mean_baseline":
        return lambda: DummyRegressor(strategy="mean")
    if name == "linear":
        return lambda: LinearRegression()
    raise ValueError(name)


def repeated_oof_eval(factory, X, y):
    """5 折 CV × N_REPEATS 重复，返回 pooled OOF 各指标。

    - ``pooled_r2``：每重复内把 5 折袋外预测拼成一条算一次 R²，再对重复求均值±std。
    - ``fold_mean_r2``：所有折的 R² 直接平均（逐折口径）。
    - ``rmse`` / ``mae``：在"全部折+全部重复"拼接的 pooled OOF 预测上计算。
    """
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    pooled_per_repeat = []
    fold_r2s = []
    all_y, all_pred = [], []

    buf_y, buf_pred = [], []
    for tr, te in rkf.split(X):
        m = factory()
        m.fit(X.iloc[tr], y[tr])
        p = np.asarray(m.predict(X.iloc[te])).ravel()
        fold_r2s.append(r2_score(y[te], p))
        buf_y.append(y[te])
        buf_pred.append(p)
        if len(buf_y) == N_SPLITS:
            pooled_per_repeat.append(r2_score(np.concatenate(buf_y), np.concatenate(buf_pred)))
            buf_y, buf_pred = [], []
        all_y.append(y[te])
        all_pred.append(p)

    yy = np.concatenate(all_y)
    pp = np.concatenate(all_pred)
    return {
        "pooled_r2": float(np.mean(pooled_per_repeat)),
        "pooled_r2_sd": float(np.std(pooled_per_repeat, ddof=1)),
        "fold_mean_r2": float(np.mean(fold_r2s)),
        "rmse": float(np.sqrt(mean_squared_error(yy, pp))),
        "mae": float(mean_absolute_error(yy, pp)),
        "n": int(len(y)),
    }


def single_fold_oof(factory, X, y):
    """单次 5 折 CV，返回每样本一个 OOF 预测（画散点图用，不重复）。"""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pred = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        m = factory()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = np.asarray(m.predict(X.iloc[te])).ravel()
    return pred


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data(DATA_CSV)
    print(f"cleaned rows={len(df)}  shear={int(df['shear_bond_strength'].notna().sum())}  "
          f"tensile={int(df['tensile_bond_strength'].notna().sum())}  "
          f"n_features={len(FEATURE_COLUMNS)}")

    models = ["xgboost", "random_forest", "bp_mlp", "linear", "mean_baseline"]
    rows = []
    scatter_preds = {}
    shap_data = {}

    for target in TARGETS:
        sub = df[df[target].notna()].reset_index(drop=True)
        X = sub[FEATURE_COLUMNS]
        y = sub[target].to_numpy()
        print(f"\n=== {target}  n={len(y)}  mean={y.mean():.2f}  std={y.std():.2f} ===")

        for name in models:
            r = repeated_oof_eval(model_factory(name), X, y)
            r.update({"target": target, "model": name})
            rows.append(r)
            flag = "  <-- primary" if name == "xgboost" else ""
            print(f"  {name:16s} pooled_R2={r['pooled_r2']:.3f}±{r['pooled_r2_sd']:.3f}  "
                  f"fold_mean={r['fold_mean_r2']:.3f}  RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}{flag}")

        # 散点：单次 5 折 OOF 预测
        pred = single_fold_oof(model_factory("xgboost"), X, y)
        scatter_preds[target] = (y, pred)

        # SHAP：全量拟合 XGBoost（论文图惯例：解释最终模型）
        xgb_full = XGBRegressor(**XGB_PARAMS)
        xgb_full.fit(X, y)
        shap_data[target] = (xgb_full, X)

    # ---- 指标表落盘 ----
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "metrics_table.csv", index=False)
    summary = {
        "data": {
            "csv": DATA_CSV,
            "cleaned_rows": int(len(df)),
            "shear_n": int(df["shear_bond_strength"].notna().sum()),
            "tensile_n": int(df["tensile_bond_strength"].notna().sum()),
        },
        "xgb_params": XGB_PARAMS,
        "cv": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "seed": SEED},
        "by_target": {
            t: {r["model"]: {k: r[k] for k in ("pooled_r2", "pooled_r2_sd", "fold_mean_r2", "rmse", "mae", "n")}
                for r in rows if r["target"] == t}
            for t in TARGETS
        },
    }
    with (OUT_DIR / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- 图：散点 + SHAP + 特征重要性 ----
    for target in TARGETS:
        y, pred = scatter_preds[target]
        tname = "shear" if target.startswith("shear") else "tensile"
        plt.figure(figsize=(5.2, 4.6))
        plt.scatter(y, pred, s=26, alpha=0.65, edgecolors="none")
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        plt.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="1:1")
        plt.xlabel(f"Measured {tname} bond strength (MPa)")
        plt.ylabel(f"Predicted {tname} bond strength (MPa)")
        plt.title(f"XGBoost — {tname} (n={len(y)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"scatter_xgboost_{target}.png", dpi=180, bbox_inches="tight")
        plt.close()

        xgb_full, X = shap_data[target]
        labels = [FEATURE_LABELS.get(c, c) for c in X.columns]
        try:
            import shap

            explainer = shap.TreeExplainer(xgb_full)
            sv = np.asarray(explainer.shap_values(X))
            if sv.ndim == 3:
                sv = sv[:, :, 0]
            # beeswarm
            shap.summary_plot(sv, X, feature_names=labels, show=False, max_display=17)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"shap_summary_{target}.png", dpi=180, bbox_inches="tight")
            plt.close()

            # 特征重要性：mean |SHAP| 条形图
            imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
            imp = imp.sort_values(ascending=True)
            imp_labeled = pd.Series(imp.values, index=[FEATURE_LABELS.get(c, c) for c in imp.index])
            plt.figure(figsize=(5.6, 5.4))
            plt.barh(imp_labeled.index, imp_labeled.values)
            plt.xlabel("mean |SHAP value|")
            plt.title(f"Feature importance — {tname} (XGBoost)")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"feature_importance_{target}.png", dpi=180, bbox_inches="tight")
            plt.close()

            native = pd.Series(xgb_full.feature_importances_, index=X.columns)
            imp_out = pd.DataFrame({
                "feature": imp.index,
                "mean_abs_shap": imp.values,
                "native_gain": native.reindex(imp.index).to_numpy(),
            }).sort_values("mean_abs_shap", ascending=False)
            imp_out.to_csv(OUT_DIR / f"shap_importance_{target}.csv", index=False)
        except Exception as exc:  # shap 兼容性兜底：用 native gain 画
            warnings.warn(f"SHAP failed for {target}: {exc}, using native gain")
            imp = pd.Series(xgb_full.feature_importances_, index=X.columns).sort_values(ascending=True)
            imp_labeled = pd.Series(imp.values, index=[FEATURE_LABELS.get(c, c) for c in imp.index])
            plt.figure(figsize=(5.6, 5.4))
            plt.barh(imp_labeled.index, imp_labeled.values)
            plt.xlabel("gain importance")
            plt.title(f"Feature importance — {tname} (XGBoost, gain)")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"feature_importance_{target}.png", dpi=180, bbox_inches="tight")
            plt.close()

    print(f"\nDONE -> {OUT_DIR}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DATA_CSV, help="文献数据 CSV 路径（旧流程演示；仓库默认合成示例）")
    ap.add_argument("--out", default=str(OUT_DIR), help="输出目录")
    args = ap.parse_args()
    DATA_CSV = args.csv
    OUT_DIR = Path(args.out)
    main()
