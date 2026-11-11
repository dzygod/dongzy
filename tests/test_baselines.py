"""\u57fa\u7ebf\u6a21\u578b\u7684\u884c\u4e3a\u6d4b\u8bd5\uff08PR#2\uff09\u3002"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uhpc_ml_app.baselines import (
    EmpiricalSqrtFcBaseline,
    MeanPredictor,
    build_baselines,
)


def _make_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.DataFrame(
        {
            "roughness_ra": np.linspace(1.0, 4.0, 12),
            "water_binder_ratio": np.linspace(0.20, 0.26, 12),
            "fiber_volume_ratio": np.linspace(1.0, 2.4, 12),
            "nc_strength_grade": np.linspace(40.0, 65.0, 12),
            "curing_age_days": np.linspace(7.0, 150.0, 12),
        }
    )
    y = pd.DataFrame(
        {
            "tensile_bond_strength": 0.4 * np.sqrt(X["nc_strength_grade"]) + 0.1,
            "shear_bond_strength": 1.2 * np.sqrt(X["nc_strength_grade"]) - 0.5,
        }
    )
    return X, y


def test_mean_predictor_outputs_training_means_for_all_rows() -> None:
    X, y = _make_df()
    m = MeanPredictor().fit(X.iloc[:8], y.iloc[:8])
    preds = m.predict(X.iloc[8:])
    assert preds.shape == (4, 2)
    # \u9884\u6d4b\u5e94\u4e25\u683c\u7b49\u4e8e\u8bad\u7ec3\u96c6\u5747\u503c\uff0c\u4e14\u6240\u6709\u884c\u76f8\u7b49
    np.testing.assert_allclose(preds[0], preds[-1])
    np.testing.assert_allclose(preds[0], y.iloc[:8].mean().to_numpy())


def test_empirical_sqrt_fc_baseline_recovers_linear_k() -> None:
    X, y = _make_df()
    # \u6784\u9020\u7684 y \u5b8c\u5168\u7b26\u5408 \u03c4 = k \u00b7 \u221af_c\uff0c\u62df\u5408\u540e\u9884\u6d4b\u5e94\u6781\u63a5\u8fd1
    baseline = EmpiricalSqrtFcBaseline(fc_column="nc_strength_grade").fit(X, y)
    preds = baseline.predict(X)
    assert preds.shape == y.shape
    # \u6bcf\u76ee\u6807 R\u00b2 \u63a5\u8fd1 1\uff080.4 \u4e0e 1.2 \u4f1a\u88ab\u6b63\u786e\u5b66\u51fa\uff09
    # \u524d\u5217\u7528 0.4 * sqrt(fc) + 0.1\uff0c\u56e0\u5e38\u6570\u9879\u9020\u6210\u5c0f\u504f\u5dee\uff0c\u4e0d\u505a 1.0 \u5e38\u6570\u7ea7\u65ad\u8a00\uff0c\u4ec5
    # \u65ad\u8a00\u9884\u6d4b\u4e0e y \u7684\u76f8\u5173\u7cfb\u6570\u66b4\u8fd1 1\u3002
    for j in range(y.shape[1]):
        r = np.corrcoef(preds[:, j], y.iloc[:, j].to_numpy())[0, 1]
        assert r > 0.99


def test_empirical_sqrt_fc_baseline_raises_without_fc_column() -> None:
    X, y = _make_df()
    X2 = X.drop(columns=["nc_strength_grade"])
    with pytest.raises(KeyError):
        EmpiricalSqrtFcBaseline(fc_column="nc_strength_grade").fit(X2, y)


def test_build_baselines_returns_three_entries() -> None:
    baselines = build_baselines()
    assert set(baselines.keys()) == {"baseline_mean", "baseline_linear", "baseline_sqrt_fc"}
