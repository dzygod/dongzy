from pathlib import Path

import pandas as pd

from uhpc_ml_app.app import run_pipeline
from uhpc_ml_app.config import TrainConfig


def test_run_pipeline_generates_outputs(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    output_dir = tmp_path / "output"

    df = pd.DataFrame(
        {
            "water_binder_ratio": [
                0.21,
                0.22,
                0.23,
                0.24,
                0.2,
                0.25,
                0.26,
                0.23,
                0.22,
                0.24,
                0.21,
                0.23,
            ],
            "fiber_volume_ratio": [1.0, 1.2, 1.5, 1.7, 1.8, 2.0, 1.9, 2.1, 2.2, 2.3, 2.4, 2.5],
            "nc_strength_grade": [40, 40, 45, 45, 50, 50, 55, 55, 60, 60, 65, 65],
            "curing_age_days": [7, 14, 14, 28, 28, 56, 56, 90, 90, 120, 120, 150],
            "interface_prep": ["SB"] * 6 + ["CH"] * 6,
            "specimen_type": ["slant_shear"] * 12,
            "tensile_bond_strength": [2.1, 2.3, 2.6, 2.9, 3.1, 3.3, 3.4, 3.7, 3.9, 4.0, 4.1, 4.3],
            "shear_bond_strength": [
                8.5,
                8.8,
                9.2,
                9.8,
                10.1,
                10.5,
                10.7,
                11.2,
                11.5,
                11.8,
                12.1,
                12.4,
            ],
        }
    )
    df.to_csv(input_csv, index=False)

    run_pipeline(input_csv, output_dir, random_state=7, config=TrainConfig.for_testing())

    assert (output_dir / "metrics_summary.csv").exists()
    assert (output_dir / "best_model_info.json").exists()
