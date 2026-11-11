from pathlib import Path

import pandas as pd

from uhpc_ml_app.config import TrainConfig
from uhpc_ml_app.data_pipeline import load_and_clean_data
from uhpc_ml_app.modeling import train_and_compare_models


def test_train_and_compare_models_returns_metrics_and_best_model(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    df = pd.DataFrame(
        {
            "water_binder_ratio": [0.21, 0.22, 0.23, 0.24, 0.2, 0.25, 0.26, 0.23, 0.22, 0.24],
            "fiber_volume_ratio": [1.0, 1.2, 1.5, 1.7, 1.8, 2.0, 1.9, 2.1, 2.2, 2.3],
            "nc_strength_grade": [40, 40, 45, 45, 50, 50, 55, 55, 60, 60],
            "curing_age_days": [7, 14, 14, 28, 28, 56, 56, 90, 90, 120],
            "interface_prep": ["SB"] * 5 + ["CH"] * 5,
            "specimen_type": ["slant_shear"] * 10,
            "tensile_bond_strength": [2.1, 2.3, 2.6, 2.9, 3.1, 3.3, 3.4, 3.7, 3.9, 4.0],
            "shear_bond_strength": [8.5, 8.8, 9.2, 9.8, 10.1, 10.5, 10.7, 11.2, 11.5, 11.8],
        }
    )
    df.to_csv(csv_path, index=False)
    cleaned = load_and_clean_data(csv_path)

    result = train_and_compare_models(cleaned, random_state=42, config=TrainConfig.for_testing())

    assert set(result["metrics"].keys()) == {"random_forest", "xgboost", "bp_neural_network"}
    assert result["best_model_name"] in {"random_forest", "xgboost", "bp_neural_network"}
    assert "tensile_bond_strength" in result["feature_importance"]
    assert "tensile_bond_strength" in result["predictions"]
