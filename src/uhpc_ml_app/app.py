from __future__ import annotations

import argparse
from pathlib import Path

from uhpc_ml_app.config import TrainConfig
from uhpc_ml_app.data_pipeline import load_and_clean_data
from uhpc_ml_app.mlflow_integration import log_result as _mlflow_log_result
from uhpc_ml_app.modeling import train_and_compare_models
from uhpc_ml_app.reporting import (
    run_shap_analysis,
    save_baseline_metrics,
    save_cv_metrics,
    save_metrics,
    save_per_target_metrics,
    save_permutation_importance,
    save_prediction_scatter,
)


def run_pipeline(
    input_csv: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    config: TrainConfig | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cleaned = load_and_clean_data(input_csv)
    result = train_and_compare_models(cleaned, random_state=random_state, config=config)

    save_metrics(result, output)
    save_per_target_metrics(result, output)
    save_cv_metrics(result, output)
    save_permutation_importance(result, output)
    save_baseline_metrics(result, output)
    save_prediction_scatter(result, output)
    run_shap_analysis(result, output)

    # 可选：把本次 run 的参数 / 指标 / 产物同步到 MLflow（需要环境变量 + mlflow 已安装）
    _mlflow_log_result(result, str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="UHPC-NC 修复界面粘结性能预测应用")
    parser.add_argument("--input", required=True, help="输入文献整合数据 CSV 路径")
    parser.add_argument("--output", default="outputs", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="启用 KNN 插值 + 高斯噪声的数据增强（仅作用于训练集）",
    )
    parser.add_argument(
        "--augment-n-synthetic",
        type=int,
        default=100,
        help="数据增强生成的合成样本数（默认 100）",
    )
    parser.add_argument(
        "--augment-noise-scale",
        type=float,
        default=0.03,
        help="数据增强高斯噪声相对特征标准差的尺度（默认 0.03）",
    )
    args = parser.parse_args()

    config = TrainConfig(
        augment=args.augment,
        augment_n_synthetic=args.augment_n_synthetic,
        augment_noise_scale=args.augment_noise_scale,
    )
    run_pipeline(args.input, args.output, args.seed, config)
    print(f"完成建模与解释性分析，结果已输出到: {args.output}")


if __name__ == "__main__":
    main()
