"""数据增强（KNN 插值 + 高斯噪声）的单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uhpc_ml_app.config import NUMERIC_FEATURE_COLUMNS
from uhpc_ml_app.data_augmentation import augment_regression


def _make_train() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    n = 12
    sa = rng.uniform(1.0, 4.0, n)
    wb = rng.uniform(0.18, 0.28, n)
    fv = rng.uniform(0.5, 2.5, n)
    nc = rng.uniform(40.0, 65.0, n)
    age = rng.uniform(7.0, 150.0, n)
    prep = np.array(["SB", "CH", "C"] * 4)
    y = 0.5 * np.sqrt(nc) + 0.3 * sa + rng.normal(0, 0.05, n)
    X = pd.DataFrame(
        {
            "water_binder_ratio": wb,
            "fiber_volume_ratio": fv,
            "nc_strength_grade": nc,
            "curing_age_days": age,
        }
    )
    # 手动 one-hot interface_prep（与 data_pipeline 一致）
    for value in ("C", "SB", "CH"):
        X[f"interface_prep__{value}"] = (prep == value).astype(int)
    return X, pd.Series(y, name="tensile_bond_strength")


def test_augment_regression_appends_synthetic_rows() -> None:
    X, y = _make_train()
    x_out, y_out = augment_regression(
        X,
        y,
        n_synthetic=20,
        numeric_columns=NUMERIC_FEATURE_COLUMNS,
        random_state=42,
    )
    assert len(x_out) == len(X) + 20
    assert len(y_out) == len(y) + 20
    # 原始样本在前，未被改动
    pd.testing.assert_frame_equal(x_out.iloc[: len(X)], X)
    pd.testing.assert_series_equal(y_out.iloc[: len(y)], y)


def test_augment_regression_preserves_onehot_integer_values() -> None:
    X, y = _make_train()
    x_out, _ = augment_regression(
        X,
        y,
        n_synthetic=30,
        numeric_columns=NUMERIC_FEATURE_COLUMNS,
        random_state=0,
    )
    onehot_cols = [c for c in X.columns if c.startswith("interface_prep__")]
    synth = x_out.iloc[len(X) :][onehot_cols]
    assert synth.isin([0, 1]).all().all()


def test_augment_regression_noop_when_n_synthetic_zero() -> None:
    X, y = _make_train()
    x_out, y_out = augment_regression(
        X, y, n_synthetic=0, numeric_columns=NUMERIC_FEATURE_COLUMNS, random_state=1
    )
    pd.testing.assert_frame_equal(x_out, X)
    pd.testing.assert_series_equal(y_out, y)


def test_augment_regression_keeps_target_name() -> None:
    X, y = _make_train()
    _, y_out = augment_regression(
        X, y, n_synthetic=5, numeric_columns=NUMERIC_FEATURE_COLUMNS, random_state=2
    )
    assert y_out.name == "tensile_bond_strength"
