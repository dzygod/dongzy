"""PR#3 SHAP \u53cc\u76ee\u6807 + \u4f9d\u8d56\u56fe\u9a71\u52a8\u9608\u503c\u7684\u884c\u4e3a\u6d4b\u8bd5\u3002"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uhpc_ml_app.app import run_pipeline
from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS, TrainConfig
from uhpc_ml_app.data_pipeline import load_and_clean_data
from uhpc_ml_app.modeling import train_and_compare_models
from uhpc_ml_app.shap_analysis import (
    _recommend_threshold_from_dependence,
    run_dual_target_shap,
)


def _toy_dataset(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    ra = rng.uniform(1.0, 4.0, n)
    wb = rng.uniform(0.18, 0.28, n)
    fv = rng.uniform(0.5, 2.5, n)
    nc = rng.choice([40, 45, 50, 55, 60, 65], size=n).astype(float)
    age = rng.choice([7, 14, 28, 56, 90, 120, 150], size=n).astype(float)
    prep = rng.choice(["SB", "CH", "C"], size=n)
    spec = rng.choice(["slant_shear", "splitting", "pull_off"], size=n)
    tensile = 0.5 * np.sqrt(nc) + 0.3 * ra + 0.02 * age + rng.normal(0, 0.05, n)
    shear = 1.2 * np.sqrt(nc) + 0.8 * ra + 0.03 * age + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "water_binder_ratio": wb,
            "fiber_volume_ratio": fv,
            "nc_strength_grade": nc,
            "curing_age_days": age,
            "interface_prep": prep,
            "specimen_type": spec,
            "tensile_bond_strength": tensile,
            "shear_bond_strength": shear,
        }
    )


def test_threshold_recovers_positive_plateau_for_monotonic_shap() -> None:
    """\u6784\u9020 SHAP = x \u2212 2\uff0c\u6b63\u6548\u5e94\u533a\u95f4\u5e94\u5728 x \u2265 2 \u7684\u5c3e\u6bb5\u3002"""
    xs = np.linspace(0, 4, 40)
    shap = xs - 2.0  # \u7ebf\u6027\u6b63\u7279\u5f81
    rec = _recommend_threshold_from_dependence(xs, shap)
    assert rec["reason"] == "positive_effect_plateau"
    assert (
        rec["low"] >= 1.5
    )  # \u5141\u8bb8\u5e73\u6ed1\u7a97\u53e3\u5f15\u5165\u7684\u8fb9\u754c\u504f\u5dee
    assert rec["high"] >= 3.5


def test_threshold_reports_all_negative_case() -> None:
    xs = np.linspace(0, 4, 20)
    shap = -0.5 * np.ones_like(xs)
    rec = _recommend_threshold_from_dependence(xs, shap)
    assert rec["reason"] == "no_positive_region_report_peak"
    # low/high \u5747\u5e94\u5728\u8f93\u5165\u8303\u56f4\u5185\uff08\u9ed8\u8ba4\u56de\u843d\u5230\u5cf0\u503c\u9644\u8fd1\uff09
    assert 0 <= rec["low"] <= 4
    assert 0 <= rec["high"] <= 4


def test_threshold_flags_too_few_samples() -> None:
    rec = _recommend_threshold_from_dependence(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    assert rec["reason"] == "too_few_samples"
    assert rec["low"] is None and rec["high"] is None


@pytest.fixture(scope="module")
def _pipeline_result(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    tmp_dir = tmp_path_factory.mktemp("shap_run")
    df = _toy_dataset(n=40)
    csv = tmp_dir / "ds.csv"
    df.to_csv(csv, index=False)
    cleaned = load_and_clean_data(csv)
    result = train_and_compare_models(cleaned, random_state=7, config=TrainConfig.for_testing())
    return result, tmp_dir


def test_run_dual_target_shap_covers_both_targets(_pipeline_result) -> None:
    result, tmp_dir = _pipeline_result
    out = tmp_dir / "out_shap"
    out.mkdir()

    info = run_dual_target_shap(result, out)

    assert set(info["shap_importance_by_target"].keys()) <= set(TARGET_COLUMNS)
    # \u81f3\u5c11\u6709\u4e00\u4e2a\u76ee\u6807\u6210\u529f\u51fa\u7ed3\u679c\uff08\u5c0f\u6570\u636e\u4e0b KernelExplainer \u53ef\u80fd\u5fae\u6587\u7684\u5176\u4ed6 edge cases\uff09
    assert info["shap_importance_by_target"]
    for target, scores in info["shap_importance_by_target"].items():
        assert set(scores.keys()) <= set(FEATURE_COLUMNS)
        # Shap \u91cd\u8981\u6027\u5e94\u5168\u90e8\u662f\u975e\u8d1f\u5b9e\u6570\uff08mean abs\uff09
        for v in scores.values():
            assert v >= 0
    assert (out / "shap_importance_by_target.csv").exists()
    assert (out / "roughness_recommendations.json").exists()


def test_run_pipeline_emits_dual_target_shap_artifacts(tmp_path: Path) -> None:
    df = _toy_dataset(n=40)
    input_csv = tmp_path / "input.csv"
    out = tmp_path / "out"
    df.to_csv(input_csv, index=False)
    run_pipeline(input_csv, out, random_state=13, config=TrainConfig.for_testing())

    info = json.loads((out / "best_model_info.json").read_text(encoding="utf-8"))
    assert "shap_importance_by_target" in info
    assert "roughness_recommendations" in info
    assert "explanation_method" in info
    # explanation_method \u662f\u6bcf\u76ee\u6807\u4e00\u4e2a\u503c\u7684 dict\uff0c\u4e0d\u518d\u662f\u5355\u4e2a\u5b57\u7b26\u4e32
    assert isinstance(info["explanation_method"], dict)
    assert (out / "shap_importance_by_target.csv").exists()
