"""UHPC-NC 修复界面粘结性能预测 — Streamlit 可视化面板。

PR#7 起面板对接 PR#2/#3/#4 产物：per-target 指标、Repeated K-Fold 95% CI、
三基线对照、跨模型 Permutation Importance、双目标 SHAP（summary + 依赖曲线）、
条件化粗糙度推荐，以及 SHAP 降级审计。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from uhpc_ml_app.app import run_pipeline
from uhpc_ml_app.config import FEATURE_COLUMNS, INTERFACE_PREP_LABELS, TrainConfig
from uhpc_ml_app.web_export import (
    build_results_zip,
    discover_plots,
    load_baseline_metrics,
    load_best_model_info,
    load_conditional_recommendations,
    load_cv_metrics,
    load_metrics_summary,
    load_per_target_metrics,
    load_permutation_importance,
    load_roughness_recommendations,
)

st.set_page_config(page_title="UHPC-NC 界面粘结预测", layout="wide")
st.title("UHPC-NC 修复界面粘结性能预测")
st.caption(
    "上传一个 CSV：可直接使用 `literature-ingest` 生成的 `literature_clean.csv`，"
    "也可按下方「输入数据字段要求」手工准备。"
    "自动完成清洗、建模、Repeated K-Fold 95% CI、基线对照、双目标 SHAP 与粗糙度工程推荐。"
)

with st.expander("输入数据字段要求", expanded=False):
    field_notes = [
        ("interface_prep", "界面处理方式（分类）", "必填（C/SB/HW/CH/M）"),
        ("specimen_type", "试件/试验类型（分类）", "必填（pull_off/slant_shear/splitting 等）"),
        ("water_binder_ratio", "水胶比 w/b", "必填"),
        ("fiber_volume_ratio", "钢纤维体积率（%）", "必填"),
        ("nc_strength_grade", "NC 基体抗压强度等级（MPa）", "必填"),
        ("curing_age_days", "养护龄期（d）", "必填"),
        ("roughness_sa", "三维面粗糙度 Sa（μm）", "可选；暂不参与建模"),
        ("tensile_bond_strength", "拉伸粘结强度（MPa）", "目标；可与抗剪只填其一"),
        ("shear_bond_strength", "抗剪粘结强度（MPa）", "目标；可与拉伸只填其一"),
    ]
    lines = ["| 列名 | 中文含义 | 必填性 |", "|---|---|---|"]
    for col, zh, req in field_notes:
        lines.append(f"| `{col}` | {zh} | {req} |")
    st.markdown("\n".join(lines))
    prep_desc = " / ".join(f"`{k}` {v}" for k, v in INTERFACE_PREP_LABELS.items())
    st.caption(
        f"`interface_prep` 取值（编号）：{prep_desc}；其余值或留空归入基准类。"
        "`specimen_type` 取值：pull_off/slant_shear/splitting/bi_shear/direct_shear/bending/mixed，"
        "二者均参与建模。其余列（如 DOI、source）不影响训练，但建议保留以便追溯与引用。"
        "物理上下界见 `src/uhpc_ml_app/schemas.py::CLEAN_SCHEMA`。"
    )

with st.sidebar:
    st.header("运行设置")
    uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
    seed = st.number_input("随机种子", min_value=1, value=42, step=1)
    st.divider()
    st.subheader("数据增强（可选）")
    augment = st.checkbox(
        "启用 KNN 插值 + 高斯噪声", value=False,
        help="KNN（K 近邻）插值：在近邻样本间做凸组合生成新样本，再叠加高斯噪声，"
        "用于扩充小样本训练集；默认关闭，仅作用于训练集。",
    )
    augment_n_synthetic = st.number_input(
        "合成样本数", min_value=0, value=100, step=10, disabled=not augment,
        help="数据增强生成的合成样本条数。",
    )
    augment_noise_scale = st.number_input(
        "噪声尺度（相对特征标准差）",
        min_value=0.0,
        value=0.03,
        step=0.01,
        format="%.3f",
        disabled=not augment,
        help="叠加高斯噪声的标准差占特征标准差的比值（0.03 即 3%）。",
    )
    st.caption("仅作用于训练集，测试集保持真实数据，保证评估诚实可比。")
    st.divider()
    run_clicked = st.button("开始训练与分析", type="primary", disabled=uploaded is None)
    st.caption(
        "⚠️ 切勿使用仓库自带的 `data/raw/literature_sample.csv` 的训练结果写论文 ——"
        "那是合成示例数据。"
    )

if uploaded is not None and not run_clicked:
    preview = pd.read_csv(uploaded).head(10)
    st.subheader("数据预览（前 10 行）")
    st.dataframe(preview, width="stretch")
    uploaded.seek(0)


def _fmt_float(x: object, ndigits: int = 4) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except (TypeError, ValueError):
        return "—"


def _render_overview_tab(output_dir: Path) -> None:
    info = load_best_model_info(output_dir)
    if not info:
        st.info("尚未生成 best_model_info.json。")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("最优模型", str(info.get("best_model", "—")))
    col2.metric("选模型依据", str(info.get("model_selection_metric", "cv_r2_mean")))
    cv = info.get("cv_metrics", {}) or {}
    mean = _fmt_float(cv.get("r2_mean"))
    low = _fmt_float(cv.get("r2_ci95_low"))
    high = _fmt_float(cv.get("r2_ci95_high"))
    col3.metric("最优模型 CV R²", mean, delta=f"95% CI [{low}, {high}]", delta_color="off")

    explanation_method = info.get("explanation_method") or {}
    failed_targets = info.get("failed_targets") or {}

    fallback_targets = [t for t, v in explanation_method.items() if v == "correlation_fallback"]
    failed_shap_targets = [t for t, v in explanation_method.items() if v == "failed"]

    # 三种降级语义分开告警：全局相关系数兜底 / 个别 target SHAP 失败 / 正常
    if fallback_targets:
        # 来自 shap_analysis._correlation_fallback：所有 target 都失败时才发生，
        # 此时整模型改用 |corr(feature, target)| 作为重要性替代，不等价于 SHAP。
        st.error(
            "SHAP 全局降级为 `correlation_fallback`：以下目标使用相关系数作为特征重要性替代，"
            "**不等价于** SHAP，请勿作为 SHAP 结论写入论文："
            + ", ".join(fallback_targets)
            + (
                "；原因：" + "; ".join(f"{k}={v}" for k, v in failed_targets.items())
                if failed_targets
                else ""
            )
        )
    elif failed_shap_targets:
        # 个别 target 的 SHAP 抛异常（其它 target 仍有有效 SHAP），UI 里明确列出失败 target
        # 及原因，便于排查（常见于 xgboost / shap 版本 base_score 序列化不兼容等）。
        st.warning(
            "SHAP 部分目标未能生成解释："
            + ", ".join(failed_shap_targets)
            + (
                "；原因："
                + "; ".join(
                    f"{k}={v}" for k, v in failed_targets.items() if k in failed_shap_targets
                )
                if failed_targets
                else ""
            )
            + "。其它目标的 SHAP 图与阈值仍可正常使用。"
        )
    elif explanation_method:
        methods = ", ".join(f"{k}={v}" for k, v in explanation_method.items())
        st.success(f"SHAP 解释器分派：{methods}")

    st.subheader("最优模型元数据")
    st.json(info)


def _render_accuracy_tab(output_dir: Path) -> None:
    metrics_df = load_metrics_summary(output_dir)
    per_target_df = load_per_target_metrics(output_dir)
    cv_df = load_cv_metrics(output_dir)
    baseline_df = load_baseline_metrics(output_dir)

    st.subheader("三模型总体指标 + Repeated K-Fold 95 % CI")
    if metrics_df.empty:
        st.info("`metrics_summary.csv` 未找到。")
    else:
        st.dataframe(metrics_df, width="stretch")
        st.download_button(
            "下载 metrics_summary.csv",
            data=(output_dir / "metrics_summary.csv").read_bytes(),
            file_name="metrics_summary.csv",
            mime="text/csv",
        )

    st.subheader("按目标拆分指标（拉伸 / 抗剪）")
    if per_target_df.empty:
        st.info("`metrics_per_target.csv` 未找到。")
    else:
        st.dataframe(per_target_df, width="stretch")
        st.download_button(
            "下载 metrics_per_target.csv",
            data=(output_dir / "metrics_per_target.csv").read_bytes(),
            file_name="metrics_per_target.csv",
            mime="text/csv",
        )

    st.subheader("Repeated K-Fold CV 明细")
    if cv_df.empty:
        st.info("`cv_metrics.csv` 未找到。")
    else:
        st.dataframe(cv_df, width="stretch")
        st.download_button(
            "下载 cv_metrics.csv",
            data=(output_dir / "cv_metrics.csv").read_bytes(),
            file_name="cv_metrics.csv",
            mime="text/csv",
        )

    st.subheader("基线对照（均值 / 线性 / τ=k·√fc）")
    st.caption("用于回答：ML 相对零学习基线与经典经验公式是否带来实质增益。")
    if baseline_df.empty:
        st.info("`baseline_metrics.csv` 未找到（可能 `include_baselines=False`）。")
    else:
        st.dataframe(baseline_df, width="stretch")
        st.download_button(
            "下载 baseline_metrics.csv",
            data=(output_dir / "baseline_metrics.csv").read_bytes(),
            file_name="baseline_metrics.csv",
            mime="text/csv",
        )


def _render_importance_tab(output_dir: Path, plots) -> None:
    pi_df = load_permutation_importance(output_dir)
    st.subheader("跨模型 Permutation Importance（R² 下降量）")
    st.caption(
        "Permutation Importance 口径**跨模型一致**，是论文中"
        "对比 RF / XGBoost / BPNN 特征贡献的推荐依据；模型内置 `feature_importances_` 跨模型不可比。"
    )
    if pi_df.empty:
        st.info("`permutation_importance.csv` 未找到。")
    else:
        models = sorted(pi_df["model"].unique())
        targets = sorted(pi_df["target"].unique())
        col1, col2 = st.columns(2)
        with col1:
            target_filter = st.selectbox("目标", targets, key="pi_target")
        with col2:
            model_filter = st.multiselect("模型", models, default=models, key="pi_models")
        filtered = pi_df[(pi_df["target"] == target_filter) & (pi_df["model"].isin(model_filter))]
        pivot = filtered.pivot_table(
            index="feature",
            columns="model",
            values="permutation_importance_mean",
            aggfunc="mean",
        ).reindex(list(FEATURE_COLUMNS))
        st.dataframe(pivot, width="stretch")
        st.bar_chart(pivot)
        st.download_button(
            "下载 permutation_importance.csv",
            data=(output_dir / "permutation_importance.csv").read_bytes(),
            file_name="permutation_importance.csv",
            mime="text/csv",
        )

    if plots.permutation_importance:
        st.subheader("Permutation Importance 条形图（按模型）")
        cols = st.columns(min(2, len(plots.permutation_importance)))
        for idx, (model, path) in enumerate(sorted(plots.permutation_importance.items())):
            with cols[idx % len(cols)]:
                st.image(str(path), caption=f"{model} — permutation importance")


def _render_scatter_tab(plots) -> None:
    st.subheader("实测值 vs 预测值 散点图")
    st.caption("红色虚线 y=x 为理想预测；点越贴近虚线说明模型泛化越好。")
    if not plots.scatter:
        st.info("未生成 `scatter_*.png` 图。")
        return
    for model in sorted(plots.scatter.keys()):
        st.markdown(f"**{model}**")
        targets = plots.scatter[model]
        cols = st.columns(min(2, len(targets)))
        for idx, target in enumerate(sorted(targets.keys())):
            with cols[idx % len(cols)]:
                st.image(str(targets[target]), caption=f"{model} — {target}")


def _render_shap_tab(output_dir: Path, plots) -> None:
    info = load_best_model_info(output_dir)
    explanation_method = info.get("explanation_method") or {} if info else {}
    shap_importance_by_target = info.get("shap_importance_by_target") or {} if info else {}

    st.subheader("SHAP 双目标可解释性")
    if not plots.shap_summary and not plots.shap_dependence:
        st.info("未生成 SHAP 图（可能是模型/库版本不兼容或样本过少）。")
        return

    targets_with_plots = sorted(set(plots.shap_summary) | set(plots.shap_dependence))
    for target in targets_with_plots:
        method = explanation_method.get(target, "—")
        tag = "🟢 SHAP" if method in {"tree", "kernel"} else f"🟠 降级（{method}）"
        with st.expander(f"{target}  ·  {tag}", expanded=True):
            if target in plots.shap_summary:
                st.image(str(plots.shap_summary[target]), caption=f"shap_summary — {target}")
            imp = shap_importance_by_target.get(target, {})
            if imp:
                imp_df = (
                    pd.DataFrame({"feature": list(imp.keys()), "mean_abs_shap": list(imp.values())})
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(imp_df, width="stretch")
            dep_map = plots.shap_dependence.get(target, {})
            if dep_map:
                st.markdown("**SHAP 依赖曲线（按粗糙度特征）**")
                cols = st.columns(min(2, len(dep_map)))
                for idx, feat in enumerate(sorted(dep_map.keys())):
                    with cols[idx % len(cols)]:
                        st.image(str(dep_map[feat]), caption=f"{target} × {feat}")


def _render_recommendation_tab(output_dir: Path) -> None:
    recs = load_roughness_recommendations(output_dir)
    cond_df = load_conditional_recommendations(output_dir)

    st.subheader("全局粗糙度推荐区间（基于 SHAP 依赖曲线最长正效应平台段）")
    st.caption(
        "阈值定义：在依赖曲线 `(x, SHAP)` 上找"
        "**最长连续 SHAP ≥ 0 区段 `[x_low, x_high]`**，作为推荐控制区间；"
        "全负时回退为最大点邻域并标注 `no_positive_region_report_peak`；样本 <6 则返回 `None`。"
    )
    recommendations = recs.get("recommendations", {})
    if not recommendations:
        st.info("`roughness_recommendations.json` 未找到或为空。")
    else:
        rows = []
        for target, by_feat in recommendations.items():
            for feat, bundle in by_feat.items():
                overall = bundle.get("overall", {})
                rows.append(
                    {
                        "target": target,
                        "feature": feat,
                        "low": overall.get("low"),
                        "high": overall.get("high"),
                        "support": overall.get("support"),
                        "reason": overall.get("reason"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        st.download_button(
            "下载 roughness_recommendations.json",
            data=json.dumps(recs, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="roughness_recommendations.json",
            mime="application/json",
        )

    st.subheader("条件化推荐（w/b × 养护龄期 3×3 桶）")
    st.caption("在耦合工况下逐桶独立计算阈值，用于给出「因地制宜」的工艺控制建议。")
    if cond_df.empty:
        st.info("`roughness_conditional_recommendations.csv` 未找到（可能样本过少未产出）。")
    else:
        st.dataframe(cond_df, width="stretch")
        st.download_button(
            "下载 roughness_conditional_recommendations.csv",
            data=(output_dir / "roughness_conditional_recommendations.csv").read_bytes(),
            file_name="roughness_conditional_recommendations.csv",
            mime="text/csv",
        )


def _render_other_plots(plots) -> None:
    if not plots.other:
        return
    st.subheader("其他产物图")
    cols = st.columns(2)
    for idx, path in enumerate(plots.other):
        with cols[idx % 2]:
            st.image(str(path), caption=path.name)


def _cleanup_previous_run() -> None:
    """删除上一次运行的临时目录（若存在），避免磁盘堆积。"""
    prev = st.session_state.get("uhpc_run_dir")
    if prev:
        shutil.rmtree(prev, ignore_errors=True)
    st.session_state["uhpc_run_dir"] = None
    st.session_state["uhpc_output_dir"] = None
    st.session_state["uhpc_pipeline_ran"] = False


# Session state 架构说明：
#   Streamlit 每次 widget 变化都会重跑脚本顶到底；如果把管道跑在 `run_clicked`
#   分支里、产物放到 `TemporaryDirectory` 上下文内，那么 Tab③ 切换 target / 模型
#   过滤器会触发 rerun、`run_clicked` 变回 False、tempdir 随上下文退出被清空，
#   导致"切 filter 就丢结果"。
#   解决：button 点击时用 `tempfile.mkdtemp` 拿一个**持久**路径，存进
#   `st.session_state`；后续 rerun 直接从 session_state 读 output_dir，
#   不再重跑管道。下次点击按钮时先清理上一次目录。
if run_clicked and uploaded is not None:
    _cleanup_previous_run()
    run_root = Path(tempfile.mkdtemp(prefix="uhpc_streamlit_"))
    # 先登记 run_root 到 session_state：即使 run_pipeline 中途抛异常，下一次点击按钮
    # 时 `_cleanup_previous_run` 仍能找到并删除这个目录，避免 /tmp 堆积孤儿 tempdir。
    st.session_state["uhpc_run_dir"] = str(run_root)
    input_path = run_root / "input.csv"
    output_dir = run_root / "outputs"
    input_path.write_bytes(uploaded.getvalue())
    try:
        with st.spinner("正在清洗、建模、交叉验证、SHAP 分析，请稍候..."):
            config = TrainConfig(
                augment=augment,
                augment_n_synthetic=int(augment_n_synthetic),
                augment_noise_scale=float(augment_noise_scale),
            )
            run_pipeline(input_path, output_dir, random_state=int(seed), config=config)
    except Exception as exc:  # noqa: BLE001 — 面向终端用户的最外层捕获：保证 UI 不白屏
        # 不手动清理；由下一次按钮点击的 `_cleanup_previous_run` 统一回收，
        # 同时保留失败目录供本次 rerun 中用户下载 / 调试（目前未实现下载，但不堵路）。
        st.session_state["uhpc_pipeline_ran"] = False
        st.error(f"训练失败：{type(exc).__name__}: {exc}")
    else:
        # 仅在管道成功完成时才标记为已跑，触发下游 Tab 渲染分支。
        st.session_state["uhpc_output_dir"] = str(output_dir)
        st.session_state["uhpc_pipeline_ran"] = True

if st.session_state.get("uhpc_pipeline_ran") and st.session_state.get("uhpc_output_dir"):
    output_dir = Path(st.session_state["uhpc_output_dir"])
    if not output_dir.exists():
        # 目录被外部清理（如操作系统重启），提示用户重跑
        st.warning("上一次运行的产物目录已失效，请重新点击「开始训练与分析」。")
        _cleanup_previous_run()
    else:
        if run_clicked:
            st.success("分析完成。产物按类别分组展示在下方 Tab。")
        else:
            st.info(
                "显示的是上一次训练的产物（切换 Tab / 过滤器不会重跑管道，"
                "点击「开始训练与分析」可用当前上传文件重新训练）。"
            )

        plots = discover_plots(output_dir)

        tabs = st.tabs(
            [
                "① 概览",
                "② 精度对比",
                "③ 特征重要性",
                "④ 预测散点",
                "⑤ SHAP 可解释性",
                "⑥ 工程推荐区间",
            ]
        )
        with tabs[0]:
            _render_overview_tab(output_dir)
        with tabs[1]:
            _render_accuracy_tab(output_dir)
        with tabs[2]:
            _render_importance_tab(output_dir, plots)
        with tabs[3]:
            _render_scatter_tab(plots)
        with tabs[4]:
            _render_shap_tab(output_dir, plots)
        with tabs[5]:
            _render_recommendation_tab(output_dir)

        _render_other_plots(plots)

        st.divider()
        zip_bytes = build_results_zip(output_dir)
        st.download_button(
            "一键下载全部产物（ZIP）",
            data=zip_bytes,
            file_name="uhpc_results.zip",
            mime="application/zip",
        )
