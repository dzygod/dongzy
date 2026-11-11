"""PR#2 \u5efa\u6a21\u9769\u9020\u7684\u6838\u5fc3\u65ad\u8a00\uff1a

- \u6bcf\u6a21\u578b\u6307\u6807\u5305\u542b per-target \u62c6\u5206\uff1b
- CV \u6307\u6807\u5b58\u5728\u4e14\u7ed3\u6784\u6b63\u786e\uff1b
- Permutation Importance \u5bf9\u4e09\u6a21\u578b\u5747\u6709\u8fd4\u56de\uff08\u5305\u62ec\u539f\u7248\u8fd4\u96f6\u7684 MLP\uff09\uff1b
- baseline_metrics \u4e0d\u4e3a\u7a7a\uff1b
- \u9884\u6d4b\u4ea7\u7269\u4e0e\u6700\u4f18\u6a21\u578b\u4fe1\u606f\u5bfc\u51fa CSV / JSON\u3002
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uhpc_ml_app.app import run_pipeline
from uhpc_ml_app.config import FEATURE_COLUMNS, TrainConfig
from uhpc_ml_app.data_pipeline import load_and_clean_data
from uhpc_ml_app.modeling import train_and_compare_models


def _toy_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 30
    sa = rng.uniform(1.0, 4.0, n)
    wb = rng.uniform(0.18, 0.28, n)
    fv = rng.uniform(0.5, 2.5, n)
    nc = rng.choice([40, 45, 50, 55, 60, 65], size=n).astype(float)
    age = rng.choice([7, 14, 28, 56, 90, 120, 150], size=n).astype(float)
    prep = rng.choice(["SB", "CH", "C"], size=n)
    spec = rng.choice(["slant_shear", "splitting", "pull_off"], size=n)
    tensile = 0.5 * np.sqrt(nc) + 0.3 * sa + 0.02 * age + rng.normal(0, 0.05, n)
    shear = 1.2 * np.sqrt(nc) + 0.8 * sa + 0.03 * age + rng.normal(0, 0.1, n)
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


def test_per_target_metrics_are_split_by_target(tmp_path: Path) -> None:
    df = _toy_dataset()
    csv = tmp_path / "ds.csv"
    df.to_csv(csv, index=False)
    cleaned = load_and_clean_data(csv)

    result = train_and_compare_models(cleaned, random_state=11, config=TrainConfig.for_testing())

    for model_name, splits in result["metrics_per_target"].items():
        assert set(splits.keys()) == {"train", "test"}
        for split_name, per_target in splits.items():
            assert set(per_target.keys()) == {"tensile_bond_strength", "shear_bond_strength"}
            for target, scores in per_target.items():
                assert set(scores.keys()) == {"r2", "rmse", "mae"}


def test_cv_metrics_contain_ci(tmp_path: Path) -> None:
    df = _toy_dataset()
    csv = tmp_path / "ds.csv"
    df.to_csv(csv, index=False)
    cleaned = load_and_clean_data(csv)

    result = train_and_compare_models(cleaned, random_state=11, config=TrainConfig.for_testing())
    for model_name, cv in result["cv_metrics"].items():
        assert {"r2_mean", "r2_ci95_low", "r2_ci95_high", "n_splits", "n_repeats"} <= set(cv.keys())
        # \u5747\u503c\u4f4d\u4e8e\u4e0a\u4e0b\u754c\u4e4b\u95f4\uff08\u6216\u5747\u4e3a NaN \u7684\u7aef\u60c5\uff0c\u5fe7\u5316\u5bb9\u5fcd\uff09
        if cv["r2_mean"] == cv["r2_mean"]:
            assert cv["r2_ci95_low"] <= cv["r2_mean"] <= cv["r2_ci95_high"]


def test_permutation_importance_exists_for_all_models(tmp_path: Path) -> None:
    df = _toy_dataset()
    csv = tmp_path / "ds.csv"
    df.to_csv(csv, index=False)
    cleaned = load_and_clean_data(csv)

    result = train_and_compare_models(cleaned, random_state=11, config=TrainConfig.for_testing())
    pim = result["permutation_importance_by_model"]
    assert set(pim.keys()) == {"random_forest", "xgboost", "bp_neural_network"}
    for model_name, per_target in pim.items():
        assert set(per_target.keys()) == {"tensile_bond_strength", "shear_bond_strength"}
        for target, feats in per_target.items():
            assert set(feats.keys()) == set(FEATURE_COLUMNS)


def test_baseline_metrics_include_three_entries(tmp_path: Path) -> None:
    df = _toy_dataset()
    csv = tmp_path / "ds.csv"
    df.to_csv(csv, index=False)
    cleaned = load_and_clean_data(csv)

    result = train_and_compare_models(cleaned, random_state=11, config=TrainConfig.for_testing())
    baselines = result["baseline_metrics"]
    assert set(baselines.keys()) == {"baseline_mean", "baseline_linear", "baseline_sqrt_fc"}
    for bname, entry in baselines.items():
        assert "by_target" in entry
        for tentry in entry["by_target"].values():
            assert "test" in tentry
            assert "cv" in tentry


def test_run_pipeline_emits_new_reporting_artifacts(tmp_path: Path) -> None:
    df = _toy_dataset()
    input_csv = tmp_path / "input.csv"
    out = tmp_path / "out"
    df.to_csv(input_csv, index=False)
    run_pipeline(input_csv, out, random_state=11, config=TrainConfig.for_testing())
    assert (out / "metrics_summary.csv").exists()
    assert (out / "metrics_per_target.csv").exists()
    assert (out / "cv_metrics.csv").exists()
    assert (out / "permutation_importance.csv").exists()
    assert (out / "baseline_metrics.csv").exists()
    assert (out / "best_model_info.json").exists()
