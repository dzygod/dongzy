from pathlib import Path

import pandas as pd

from uhpc_ml_app.data_pipeline import load_and_clean_data


def test_load_and_clean_data_handles_missing_and_outliers(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "water_binder_ratio": [0.2, 0.25, 0.24, None],
            "fiber_volume_ratio": [1.5, 2.0, 1.8, 1.7],
            "nc_strength_grade": [40, 45, 500.0, 42],
            "curing_age_days": [7, 14, 28, 28],
            "interface_prep": ["SB", "CH", "C", "SB"],
            "specimen_type": ["slant_shear", "slant_shear", "splitting", "slant_shear"],
            "tensile_bond_strength": [2.5, 3.2, 20.0, 2.9],
            "shear_bond_strength": [9.8, 11.1, 80.0, 10.2],
        }
    )
    df.to_csv(csv_path, index=False)

    # 此测试样本量 <8，pandera CLEAN_SCHEMA 的 min_sample_count 会拦截；
    # 这里关闭 schema 校验只验证清洗路径的数值行为。
    # 真正的 schema 行为由 test_schemas.py 覆盖。
    cleaned = load_and_clean_data(csv_path, validate=False)

    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) < len(df)
    assert "roughness_sa" not in cleaned.columns
    assert "interface_prep__SB" in cleaned.columns
    assert "specimen_type__slant_shear" in cleaned.columns
