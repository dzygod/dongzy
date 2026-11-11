"""\u7528\u4e8e\u4e0e RF / XGBoost / BP \u5bf9\u7167\u7684\u57fa\u7ebf\u6a21\u578b\u3002

\u8bc4\u5ba1\u62a5\u544a\u6307\u51fa\uff0c\u5728 ~100 \u884c\u6837\u672c\u4e0a\u65e0\u57fa\u7ebf\u5bf9\u7167\uff0c\u65e0\u6cd5\u8bc1\u660e ML \u6a21\u578b
\u771f\u7684\u80dc\u8fc7\u7ecf\u9a8c\u4e0e\u5747\u503c\u9884\u6d4b\u3002\u672c\u6587\u4ef6\u63d0\u4f9b\u4e09\u4e2a\u7ef4\u5ea6\u7684\u57fa\u7ebf\uff1a

1. ``MeanPredictor``\uff1a\u603b\u662f\u8f93\u51fa\u8bad\u7ec3\u96c6\u5747\u503c\uff0c\u4ee3\u8868\u201c\u4e0d\u5b66\u4e60\u201d\u7684\u4e0b\u754c\u3002
2. ``LinearRegression``\uff1a\u7ebf\u6027\u56de\u5f52\uff0c\u4ee3\u8868\u7b80\u5355\u53ef\u89e3\u91ca\u57fa\u7ebf\u3002
3. ``EmpiricalSqrtFcBaseline``\uff1a\u571f\u6728\u8bba\u6587\u4e2d\u5e38\u7528\u7684\u7ecf\u9a8c\u5f0f
   \u03c4 = k \u00b7 \u221a(f_c)\uff0ck \u7531\u6700\u5c0f\u4e8c\u4e58\u5728\u8bad\u7ec3\u96c6\u4e0a\u62df\u5408\u3002

\u6240\u6709\u57fa\u7ebf\u90fd\u9075\u5faa sklearn ``fit(X, y) -> self`` / ``predict(X) -> np.ndarray``
\u7ea6\u5b9a\uff0c\u4ee5\u4fbf\u76f4\u63a5\u590d\u7528 GridSearchCV / cross_val_score \u4ee5\u5916\u7684\u8bc4\u4f30\u51fd\u6570\u3002
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class MeanPredictor(BaseEstimator, RegressorMixin):
    """\u7531\u8bad\u7ec3\u96c6\u5404\u76ee\u6807\u5747\u503c\u6784\u6210\u7684\u5e38\u6570\u9884\u6d4b\u5668\u3002"""

    def __init__(self) -> None:
        self.means_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray) -> MeanPredictor:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self.means_ = y_arr.mean(axis=0)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("MeanPredictor must be fitted before predict().")
        n = len(X)
        out = np.tile(self.means_, (n, 1))
        return out


class EmpiricalSqrtFcBaseline(BaseEstimator, RegressorMixin):
    """\u201c\u03c4 = k \u00b7 \u221a(f_c)\u201d \u7ecf\u9a8c\u516c\u5f0f\u57fa\u7ebf\u3002

    - ``fc_column``\uff1a\u4f5c\u4e3a f_c \u7684\u7279\u5f81\u5217\u540d\uff0c\u9ed8\u8ba4 ``nc_strength_grade``\uff08\u7eaf\u6570\u503c\u7b49\u7ea7\uff0c
      \u4f5c\u4e3a MPa \u7684\u8fd1\u4f3c\uff09\u3002
    - \u5bf9\u6bcf\u4e2a\u76ee\u6807\u5355\u72ec\u62df\u5408\u4e00\u4e2a k\uff0c\u4ee5\u6700\u5c0f\u4e8c\u4e58\u5728\u8bad\u7ec3\u96c6\u4e0a\u6c42\u89e3\u3002
    - \u8f93\u51fa \u0177 = k \u00b7 \u221a(f_c)\uff0c\u5e76\u5bf9 f_c \u975e\u6b63\u7684\u7279\u6b8a\u503c\u9632\u62a4\u3002

    \u8fd9\u4e0d\u662f\u90a3\u79cd "state-of-the-art" \u516c\u5f0f\uff0c\u4ec5\u4f5c\u4e3a "\u6bd4\u7eaf\u5747\u503c\u5f3a\u3001
    \u6bd4\u7ebf\u6027\u56de\u5f52\u5f31" \u7684\u884c\u4e1a\u89c4\u8303\u5f0f\u5bf9\u7167\u70b9\u3002
    """

    def __init__(self, fc_column: str = "nc_strength_grade") -> None:
        self.fc_column = fc_column
        self.k_: np.ndarray | None = None
        self.target_names_: list[str] | None = None

    def _select_fc(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.fc_column not in X.columns:
                raise KeyError(
                    f"fc_column={self.fc_column} not found in X columns: {list(X.columns)}"
                )
            fc = X[self.fc_column].to_numpy(dtype=float)
        else:
            raise TypeError(
                "EmpiricalSqrtFcBaseline requires X as pandas DataFrame so fc column can be selected."
            )
        return np.clip(fc, a_min=1e-6, a_max=None)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray) -> EmpiricalSqrtFcBaseline:
        fc = self._select_fc(X)
        sqrt_fc = np.sqrt(fc)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        # LS closed form for y = k * sqrt_fc  ->  k = (sqrt_fc \cdot y) / (sqrt_fc \cdot sqrt_fc)
        denom = float(np.dot(sqrt_fc, sqrt_fc))
        if denom == 0.0:
            self.k_ = np.zeros(y_arr.shape[1])
        else:
            self.k_ = (sqrt_fc @ y_arr) / denom  # shape: (n_targets,)
        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.k_ is None:
            raise RuntimeError("EmpiricalSqrtFcBaseline must be fitted before predict().")
        fc = self._select_fc(X)
        sqrt_fc = np.sqrt(fc)
        return np.outer(sqrt_fc, self.k_)


def build_baselines() -> dict[str, BaseEstimator]:
    """\u6784\u9020\u672c\u9879\u76ee\u7edf\u4e00\u4f7f\u7528\u7684\u57fa\u7ebf\u6a21\u578b\u96c6\u3002"""
    return {
        "baseline_mean": MeanPredictor(),
        "baseline_linear": LinearRegression(),
        "baseline_sqrt_fc": EmpiricalSqrtFcBaseline(fc_column="nc_strength_grade"),
    }


__all__ = [
    "MeanPredictor",
    "EmpiricalSqrtFcBaseline",
    "build_baselines",
]
