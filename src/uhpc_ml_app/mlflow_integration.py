"""\u53ef\u9009\u7684 MLflow \u96c6\u6210\uff1a\u5b9e\u9a8c\u8ddf\u8e2a\u4e0d\u5f3a\u5236\uff0c\u4ec5\u5728\u5b89\u88c5\u4e86 ``mlflow`` \u4e14\u8bbe\u7f6e\u4e86\u73af\u5883\u53d8\u91cf\u65f6\u542f\u7528\u3002

\u7528\u6cd5\uff1a

1. ``pip install uhpc-ml-app[mlflow]``\uff08\u6216\u76f4\u63a5 ``pip install mlflow``\uff09\uff1b
2. \u8bbe\u7f6e ``MLFLOW_TRACKING_URI`` \u548c\uff08\u53ef\u9009\uff09``MLFLOW_EXPERIMENT_NAME``\uff1b
3. ``run_pipeline(...)`` \u4f1a\u5728\u8fd0\u884c\u7ed3\u675f\u540e\u81ea\u52a8\u8c03\u7528 :func:`log_result`\u3002

\u5982\u679c\u7f3a\u5c11\u4efb\u4f55\u4e00\u4e2a\u6761\u4ef6\uff0c\u8be5\u6a21\u5757\u4f1a\u6d3e\u7ecf\u7981\u7528\u800c\u4e0d\u4f1a\u629b\u5f02\u5e38\uff0c
\u907f\u514d\u5728\u6ca1\u6709\u6784\u5efa tracking server \u7684\u7b80\u5355\u8bba\u6587\u590d\u73b0\u573a\u666f\u4e2d\u5f3a\u5236\u4f9d\u8d56\u3002
"""

from __future__ import annotations

import contextlib
import os
from typing import Any


def _mlflow_module() -> Any | None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        import mlflow  # type: ignore[import-not-found]

        return mlflow
    except Exception:
        return None


def log_result(result: dict, output_dir: str) -> bool:
    """\u5c06\u5173\u952e\u6307\u6807\u4e0e\u4ea7\u7269\u76ee\u5f55\u5199\u5165 MLflow\uff1b\u8fd4\u56de\u662f\u5426\u771f\u7684\u6210\u529f\u8bb0\u5f55\u3002

    \u6574\u4e2a\u5199\u5165\u6d41\u7a0b\u90fd\u88ab try/except \u5305\u88f9\uff1a``MLFLOW_TRACKING_URI`` \u4e07\u4e00\u6307\u5411\u4e0d\u53ef\u8fbe
    \u6216\u8ba4\u8bc1\u5931\u8d25\u7684 server\uff08start_run / log_param / log_metric \u90fd\u4f1a\u629b\uff09\uff0c
    \u4e5f\u4e0d\u4f1a\u628a\u4e3b\u7ba1\u9053\u62d6\u5d29\u3002
    """
    mlflow = _mlflow_module()
    if mlflow is None:
        return False

    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "uhpc-ml-app")
    best_name = result.get("best_model_name", "unknown")

    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"pipeline_{best_name}"):
            # \u6a21\u578b\u9009\u62e9\u6307\u6807\uff08CV-R\u00b2\u4f18\u4e8e test-R\u00b2\uff09
            metric_key = result.get("model_selection_metric", "cv_r2_mean")
            mlflow.log_param("best_model", best_name)
            mlflow.log_param("model_selection_metric", metric_key)

            # overall \u6307\u6807
            for model, metrics in result.get("metrics", {}).items():
                for split in ("train", "test"):
                    for stat_name, value in metrics.get(split, {}).items():
                        mlflow.log_metric(f"{model}.{split}.{stat_name}", float(value))
            # cv \u6307\u6807
            for model, cv in result.get("cv_metrics", {}).items():
                for stat_name in ("r2_mean", "r2_ci95_low", "r2_ci95_high"):
                    v = cv.get(stat_name)
                    if v is not None:
                        mlflow.log_metric(f"{model}.cv.{stat_name}", float(v))
            # \u4ea7\u7269\u76ee\u5f55
            with contextlib.suppress(Exception):
                mlflow.log_artifacts(output_dir)
    except Exception as exc:
        # tracking server \u4e0d\u53ef\u8fbe / \u8ba4\u8bc1\u5931\u8d25 / \u6587\u4ef6\u5199\u5165\u5f02\u5e38 \u2014\u2014 \u90fd\u4e0d\u80fd\u629b\u5230\u4e3b\u7ba1\u9053
        import warnings

        warnings.warn(
            f"MLflow logging failed, continuing without tracking: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


__all__ = ["log_result"]
