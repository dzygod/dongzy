"""`web_export` 读取层的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_build_results_zip_contains_generated_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_summary.csv").write_text("model,test_r2\nrf,0.9\n", encoding="utf-8")
    (out_dir / "plot.png").write_bytes(b"fakepng")

    zip_bytes = build_results_zip(out_dir)

    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0


def test_loaders_return_empty_when_missing(tmp_path: Path) -> None:
    # 空目录下，所有 CSV loader 应返回空 DataFrame，JSON loader 应返回空 dict，
    # 绝不能因为文件缺失抛异常（界面要能优雅降级）。
    out_dir = tmp_path / "empty"
    out_dir.mkdir()

    assert load_metrics_summary(out_dir).empty
    assert load_per_target_metrics(out_dir).empty
    assert load_cv_metrics(out_dir).empty
    assert load_baseline_metrics(out_dir).empty
    assert load_permutation_importance(out_dir).empty
    assert load_conditional_recommendations(out_dir).empty

    assert load_best_model_info(out_dir) == {}
    assert load_roughness_recommendations(out_dir) == {}


def test_loaders_round_trip(tmp_path: Path) -> None:
    out_dir = tmp_path / "present"
    out_dir.mkdir()

    _write_csv(out_dir / "metrics_summary.csv", pd.DataFrame({"model": ["rf"], "test_r2": [0.9]}))
    _write_csv(
        out_dir / "metrics_per_target.csv",
        pd.DataFrame(
            {
                "model": ["rf", "rf"],
                "split": ["test", "test"],
                "target": ["tensile_bond_strength", "shear_bond_strength"],
                "r2": [0.88, 0.81],
            }
        ),
    )
    _write_csv(
        out_dir / "cv_metrics.csv",
        pd.DataFrame(
            {
                "model": ["rf"],
                "r2_mean": [0.85],
                "r2_ci95_low": [0.80],
                "r2_ci95_high": [0.90],
            }
        ),
    )
    _write_csv(
        out_dir / "baseline_metrics.csv",
        pd.DataFrame({"model": ["baseline_mean"], "test_r2": [-0.01]}),
    )
    _write_csv(
        out_dir / "permutation_importance.csv",
        pd.DataFrame(
            {
                "model": ["rf"],
                "target": ["tensile_bond_strength"],
                "feature": ["roughness_sa"],
                "permutation_importance_mean": [0.12],
            }
        ),
    )
    _write_csv(
        out_dir / "roughness_conditional_recommendations.csv",
        pd.DataFrame(
            {
                "target": ["tensile_bond_strength"],
                "feature": ["roughness_sa"],
                "condition": ["water_binder_ratio"],
                "bucket": ["low"],
                "low": [1.5],
                "high": [3.0],
            }
        ),
    )
    _write_json(out_dir / "best_model_info.json", {"best_model": "rf"})
    _write_json(
        out_dir / "roughness_recommendations.json",
        {
            "best_model": "rf",
            "recommendations": {
                "tensile_bond_strength": {
                    "roughness_sa": {
                        "overall": {"low": 1.5, "high": 3.0, "reason": "positive_effect_plateau"},
                        "conditional": {},
                    }
                }
            },
        },
    )

    assert load_metrics_summary(out_dir).iloc[0]["model"] == "rf"
    assert len(load_per_target_metrics(out_dir)) == 2
    cv = load_cv_metrics(out_dir).iloc[0]
    assert cv["r2_ci95_low"] == 0.80 and cv["r2_ci95_high"] == 0.90
    assert load_baseline_metrics(out_dir).iloc[0]["model"] == "baseline_mean"
    assert load_permutation_importance(out_dir).iloc[0]["feature"] == "roughness_sa"
    cond = load_conditional_recommendations(out_dir)
    assert cond.iloc[0]["bucket"] == "low"

    info = load_best_model_info(out_dir)
    assert info["best_model"] == "rf"
    recs = load_roughness_recommendations(out_dir)
    assert (
        recs["recommendations"]["tensile_bond_strength"]["roughness_sa"]["overall"]["reason"]
        == "positive_effect_plateau"
    )


def test_discover_plots_groups_by_convention(tmp_path: Path) -> None:
    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    # 使用仓库里约定的命名生成一批空 PNG
    fnames = [
        "scatter_random_forest_tensile_bond_strength.png",
        "scatter_random_forest_shear_bond_strength.png",
        "scatter_xgboost_tensile_bond_strength.png",
        "shap_summary_tensile_bond_strength.png",
        "shap_summary_shear_bond_strength.png",
        "shap_dependence_tensile_bond_strength_water_binder_ratio.png",
        "shap_dependence_shear_bond_strength_water_binder_ratio.png",
        "permutation_importance_random_forest.png",
        "permutation_importance_xgboost.png",
        "something_else.png",
    ]
    for name in fnames:
        (out_dir / name).write_bytes(b"fake")

    plots = discover_plots(out_dir)

    assert set(plots.scatter.keys()) == {"random_forest", "xgboost"}
    assert set(plots.scatter["random_forest"].keys()) == {
        "tensile_bond_strength",
        "shear_bond_strength",
    }
    assert set(plots.shap_summary.keys()) == {"tensile_bond_strength", "shear_bond_strength"}
    assert plots.shap_dependence["tensile_bond_strength"]["water_binder_ratio"].name.endswith(".png")
    assert plots.shap_dependence["shear_bond_strength"]["water_binder_ratio"].name.endswith(".png")
    assert set(plots.permutation_importance.keys()) == {"random_forest", "xgboost"}
    assert [p.name for p in plots.other] == ["something_else.png"]


def test_discover_plots_handles_empty_dir(tmp_path: Path) -> None:
    plots = discover_plots(tmp_path)
    assert plots.scatter == {}
    assert plots.shap_summary == {}
    assert plots.shap_dependence == {}
    assert plots.permutation_importance == {}
    assert plots.other == []
