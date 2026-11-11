# -*- coding: utf-8 -*-
"""小论文1 · 跨研究验证 v4（按审稿意见修正 v3 的 IQR 数据泄漏）。

审稿意见（样本筛选与验证边界）：
    v3 在分折前用**整库分位数**做 IQR 删行，随后才在折内计算裁剪界限；因此"预处理全部
    仅在训练折内完成"不成立——整库删行会把测试集信息泄漏进"哪些样本被纳入"的决策。

本版修正（两条都做）：
    1) 主分析诚实改名为 ``global_prescreen``：明确其 LOGO 指标是"预先筛定的 N 条记录上的
       条件性能"，不再声称"完全折内预处理"；
    2) 新增敏感性 ``train_only_inclusion``：**完全去掉全局删行**，改为每个 LOGO 折内用
       **训练研究的分位数**确定 IQR 纳入界限，据此删除训练/测试中越界行（测试行用训练界限
       筛选，测试信息不进界限）。这是无泄漏口径，作为主结论的稳健性检验。

其余沿用 v3：主分析不按龄期删行（龄期仅在物理界 [0.5,400]）；敏感性保留 v2 的龄期限制。

跑法（真实数据，仓库不附带，需按 data_protocol 整理或向通讯作者请求）：
    python scripts/paper1_group_validation_v4.py --csv <path>/uhpc_nc_bond_literature_data_v6.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR_DEFAULT = _SCRIPT_DIR.parent / "outputs" / "run-group-v4"

NUMERIC = ["water_binder_ratio", "fiber_volume_ratio", "nc_strength_grade", "curing_age_days"]
IQR_FEATURES = ["water_binder_ratio", "fiber_volume_ratio", "nc_strength_grade"]  # 不含龄期
SPARSE = ["roughness_mm"]
PREP = ["C", "SB", "HW", "CH", "M"]
ST = ["pull_off", "slant_shear", "splitting", "bi_shear", "direct_shear", "direct_tension"]
TARGETS = ["tensile_bond_strength", "shear_bond_strength"]
FEATURES = NUMERIC + SPARSE + [f"interface_prep__{v}" for v in PREP] + [f"specimen_type__{v}" for v in ST]
METHOD_FEATURES = [f"specimen_type__{v}" for v in ST]

XGB = dict(n_estimators=300, learning_rate=0.02, max_depth=5, subsample=0.8,
           colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=42)
RF = dict(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=1)
SEED = 42


def build_cohort(csv, drop_age=True, global_iqr=True):
    """构建 cohort。

    - ``drop_age``：True → v2 敏感性（龄期也参与 IQR 删行，得到 375 条）；False → 主分析（不删龄期）。
    - ``global_iqr``：True → 整库分位数 IQR 删行（**审稿指出这会泄漏测试集信息**，仅作为
      "预先筛定的条件性能"口径保留）；False → 不做全局删行，纳入界限交给折内训练确定。
    """
    df = pd.read_csv(csv)
    meta_cols = ["record_id", "doi", "first_author", "year", "specimen_type", "confidence"]
    w = df.copy()
    counts = {"raw": len(w)}
    w["interface_prep"] = w["interface_prep"].astype(str).str.strip().str.upper().replace(
        ["", "NAN", "NONE", "NULL", "UNKNOWN", "N/A", "NA", "-"], None)
    w["specimen_type"] = w["specimen_type"].astype(str).str.strip().str.lower()
    for c in NUMERIC + SPARSE + TARGETS + ["confidence"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[w[TARGETS].notna().any(axis=1)]
    counts["after_target"] = len(w)
    w = w.dropna(subset=NUMERIC)
    counts["after_complete_numeric"] = len(w)
    if global_iqr:
        iqr_feats = NUMERIC if drop_age else IQR_FEATURES
        for c in iqr_feats:
            q1, q3 = w[c].quantile(.25), w[c].quantile(.75); iqr = q3 - q1
            if iqr < 1e-12:
                continue
            w = w[(w[c] >= q1 - 1.5 * iqr) & (w[c] <= q3 + 1.5 * iqr)]
    counts["after_global_iqr" if global_iqr else "before_fold_iqr"] = len(w)
    n_unrated = int(w["confidence"].isna().sum())
    w = w[w["confidence"].notna()]
    counts["after_confidence"] = len(w)
    counts["unrated_dropped"] = n_unrated
    w = w.reset_index(drop=True)
    meta = w[meta_cols].copy()
    w[SPARSE[0]] = w[SPARSE[0]].fillna(-1.0)
    prep = w["interface_prep"]; st = w["specimen_type"]
    for v in PREP:
        w[f"interface_prep__{v}"] = (prep == v).astype(int)
    for v in ST:
        w[f"specimen_type__{v}"] = (st == v).astype(int)
    return w, meta, counts


def clip_fold(Xtr, Xte, cols=IQR_FEATURES):
    """折内**裁剪**（不删行）：用训练折分位数 clip 训练/测试数值特征。"""
    a, b = Xtr.copy(), Xte.copy()
    for c in cols:
        q1, q3 = a[c].quantile(.25), a[c].quantile(.75); iqr = q3 - q1
        if iqr >= 1e-12:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            a[c] = a[c].clip(lo, hi); b[c] = b[c].clip(lo, hi)
    return a, b


def _train_iqr_mask(Xtr, Xte, cols=IQR_FEATURES):
    """用**训练折**分位数确定 IQR 纳入界限，返回训练/测试的保留布尔掩码（删行而非裁剪）。

    测试行用训练界限筛选，测试信息不进界限 —— 这是无泄漏口径。
    """
    keep_tr = pd.Series(True, index=Xtr.index)
    keep_te = pd.Series(True, index=Xte.index)
    for c in cols:
        q1, q3 = Xtr[c].quantile(.25), Xtr[c].quantile(.75); iqr = q3 - q1
        if iqr < 1e-12:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep_tr &= (Xtr[c] >= lo) & (Xtr[c] <= hi)
        keep_te &= (Xte[c] >= lo) & (Xte[c] <= hi)
    return keep_tr.to_numpy(), keep_te.to_numpy()


def _mask_nan(y, p):
    m = ~np.isnan(np.asarray(p, dtype=float))
    return np.asarray(y, dtype=float)[m], np.asarray(p, dtype=float)[m]


def r2_rmse_mae(y, p):
    yv, pv = _mask_nan(y, p)
    return (float(r2_score(yv, pv)), float(np.sqrt(mean_squared_error(yv, pv))),
            float(mean_absolute_error(yv, pv)))


def logo_eval(X, y, groups, specimen_type):
    """主分析：全局预筛后的 LOGO（折内裁剪）。"""
    g = np.asarray(groups); st = np.asarray(specimen_type); n = len(y)
    oof_xgb = np.full(n, np.nan); oof_lin = np.full(n, np.nan)
    oof_mean = np.full(n, np.nan); oof_mmean = np.full(n, np.nan)
    oof_lin_method = np.full(n, np.nan)  # 仅试件类型哑变量
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups=g):
        a, b = clip_fold(X.iloc[tr], X.iloc[te])
        oof_xgb[te] = XGBRegressor(**XGB).fit(a, y[tr]).predict(b)
        oof_lin[te] = LinearRegression().fit(a, y[tr]).predict(b)
        lm = LinearRegression().fit(a[METHOD_FEATURES], y[tr])
        oof_lin_method[te] = lm.predict(b[METHOD_FEATURES])
        oof_mean[te] = y[tr].mean()
        for i in te:
            same = st[tr] == st[i]
            oof_mmean[i] = y[tr][same].mean() if same.any() else y[tr].mean()
    return oof_xgb, oof_lin, oof_mean, oof_mmean, oof_lin_method


def logo_eval_train_only_iqr(X, y, groups, specimen_type):
    """敏感性（审稿核心修正）：IQR 纳入界限完全由训练研究确定，无全局删行。

    每个 LOGO 折内：用训练研究分位数定 IQR 界限 → 删训练/测试越界行 → 训练 → 预测。
    被训练界限排除的测试行不参与评估（OOF 记 NaN）。
    """
    g = np.asarray(groups); st = np.asarray(specimen_type); n = len(y)
    oof_xgb = np.full(n, np.nan); oof_lin = np.full(n, np.nan)
    oof_mean = np.full(n, np.nan); oof_mmean = np.full(n, np.nan)
    oof_lin_method = np.full(n, np.nan)
    n_eval = 0
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups=g):
        keep_tr, keep_te = _train_iqr_mask(X.iloc[tr], X.iloc[te])
        tr2 = tr[keep_tr]; te2 = te[keep_te]
        if len(tr2) < 2 or len(te2) == 0:
            continue
        a = X.iloc[tr2]; b = X.iloc[te2]; ytr = y[tr2]
        n_eval += len(te2)
        oof_xgb[te2] = XGBRegressor(**XGB).fit(a, ytr).predict(b)
        oof_lin[te2] = LinearRegression().fit(a, ytr).predict(b)
        lm = LinearRegression().fit(a[METHOD_FEATURES], ytr)
        oof_lin_method[te2] = lm.predict(b[METHOD_FEATURES])
        oof_mean[te2] = ytr.mean()
        for i in te2:
            same = st[tr2] == st[i]
            oof_mmean[i] = ytr[same].mean() if same.any() else ytr.mean()
    return oof_xgb, oof_lin, oof_mean, oof_mmean, oof_lin_method, n_eval


def within_random5(X, y, models=("xgb", "rf")):
    out = {}
    for name in models:
        pool_r2, pool_rmse, pool_mae = [], [], []
        for seed in range(30):
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            yy, pp = [], []
            for tr, te in kf.split(X):
                a, b = clip_fold(X.iloc[tr], X.iloc[te])
                if name == "xgb":
                    m = XGBRegressor(**XGB)
                else:
                    m = RandomForestRegressor(**RF)
                m.fit(a, y[tr]); yy.append(y[te]); pp.append(m.predict(b))
            yc = np.concatenate(yy); pc = np.concatenate(pp)
            pool_r2.append(r2_score(yc, pc))
            pool_rmse.append(np.sqrt(mean_squared_error(yc, pc)))
            pool_mae.append(mean_absolute_error(yc, pc))
        out[name] = dict(r2=float(np.mean(pool_r2)), r2_sd=float(np.std(pool_r2, ddof=1)),
                         rmse=float(np.mean(pool_rmse)), mae=float(np.mean(pool_mae)))
    return out


def _run_target(w, meta, target, tag, inclusion):
    """对单个目标跑 LOGO（按 inclusion 选主口径或训练内口径），打印并返回报告条目。"""
    tname = "shear" if target.startswith("shear") else "tensile"
    sub = w[w[target].notna()]
    m = meta.loc[sub.index].reset_index(drop=True)
    sub = sub.reset_index(drop=True)
    X = sub[FEATURES]; y = sub[target].to_numpy()
    groups = m["doi"]; st = m["specimen_type"]
    print(f"\n--- [{tag}] {tname} n={len(y)} studies={groups.nunique()} "
          f"mean={y.mean():.2f} std={y.std():.2f} ---")

    if inclusion == "train_only":
        oof_xgb, oof_lin, oof_mean, oof_mmean, oof_lin_method, n_eval = \
            logo_eval_train_only_iqr(X, y, groups, st)
        print(f"  （训练界限纳入后实际评估 {n_eval}/{len(y)} 条，其余被训练 IQR 界限排除）")
    else:
        oof_xgb, oof_lin, oof_mean, oof_mmean, oof_lin_method = logo_eval(X, y, groups, st)

    r = dict(
        logo_xgb=r2_rmse_mae(y, oof_xgb), logo_lin=r2_rmse_mae(y, oof_lin),
        logo_mean=r2_rmse_mae(y, oof_mean), logo_mmean=r2_rmse_mae(y, oof_mmean),
        logo_lin_method=r2_rmse_mae(y, oof_lin_method),
    )
    wr = within_random5(X, y)
    print(f"  [同研究随机5折×30] XGB R²={wr['xgb']['r2']:.3f}±{wr['xgb']['r2_sd']:.3f} RMSE={wr['xgb']['rmse']:.3f} MAE={wr['xgb']['mae']:.3f}")
    print(f"  [同研究随机5折×30] RF  R²={wr['rf']['r2']:.3f}")
    print(f"  [LOGO] XGBoost      R²={r['logo_xgb'][0]:.3f} RMSE={r['logo_xgb'][1]:.3f} MAE={r['logo_xgb'][2]:.3f}")
    print(f"  [LOGO] 线性(全部)   R²={r['logo_lin'][0]:.3f} RMSE={r['logo_lin'][1]:.3f}")
    print(f"  [LOGO] 线性(仅方法) R²={r['logo_lin_method'][0]:.3f}")
    print(f"  [LOGO] 方法均值     R²={r['logo_mmean'][0]:.3f}")
    print(f"  [LOGO] 全局均值     R²={r['logo_mean'][0]:.3f}")

    entry = dict(
        n=int(len(y)), studies=int(groups.nunique()),
        inclusion=inclusion,
        within=wr, logo={k: dict(zip(['r2', 'rmse', 'mae'], v)) for k, v in r.items()},
    )
    if inclusion == "train_only":
        entry["n_eval_after_train_iqr"] = n_eval
    if tname == "shear":
        pm = {}
        for meth in ["slant_shear", "bi_shear", "direct_shear"]:
            msk = (st == meth).to_numpy()
            valid = msk & ~np.isnan(oof_xgb)
            if valid.sum() == 0:
                continue
            rr = r2_rmse_mae(y[valid], oof_xgb[valid])
            pm[meth] = dict(n=int(valid.sum()), rmse=round(rr[1], 2), mae=round(rr[2], 2), r2=round(rr[0], 3))
            print(f"    {meth:16s} n={valid.sum():3d} RMSE={rr[1]:.2f} MAE={rr[2]:.2f} R²={rr[0]:.3f}")
        entry["per_method"] = pm
    return entry


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1) 主分析 + 龄期敏感性：全局 IQR 预筛口径（诚实标注为"预先筛定的条件性能"）
    for drop_age in (False, True):
        tag = "main_global_prescreen" if not drop_age else "sensitivity_age_restricted"
        w, meta, counts = build_cohort(DATA_CSV, drop_age, global_iqr=True)
        print(f"\n===== cohort[{tag}] (global_iqr) =====")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        print(f"  研究数: {meta['doi'].nunique()}")
        for target in TARGETS:
            report[f"{'shear' if target.startswith('shear') else 'tensile'}_{tag}"] = \
                _run_target(w, meta, target, tag, "global_prescreen")

    # 2) 敏感性：训练研究确定纳入界限（审稿意见核心修正，无泄漏口径）
    tag = "sensitivity_train_only_inclusion"
    w, meta, counts = build_cohort(DATA_CSV, False, global_iqr=False)
    print(f"\n===== cohort[{tag}] (no global_iqr) =====")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  研究数: {meta['doi'].nunique()}")
    for target in TARGETS:
        report[f"{'shear' if target.startswith('shear') else 'tensile'}_{tag}"] = \
            _run_target(w, meta, target, tag, "train_only")

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDONE -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="小论文1 跨研究验证 v4（修正 IQR 数据泄漏）")
    ap.add_argument("--csv", required=True, help="真实文献数据 CSV（uhpc_nc_bond_literature_data_v6.csv）")
    ap.add_argument("--out", default=str(OUT_DIR_DEFAULT), help="输出目录")
    args = ap.parse_args()
    DATA_CSV = args.csv
    OUT_DIR = Path(args.out)
    main()
