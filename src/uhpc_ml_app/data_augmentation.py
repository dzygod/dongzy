"""小样本回归数据增强：KNN 插值 + 高斯噪声。

科研文献整合数据通常只有数十到百余行，直接训练树/神经网络模型容易过拟合、
且在特征空间覆盖不足。本模块提供轻量、可解释、无额外依赖的合成策略，**只作用于训练集**：

1. 对每条真实训练样本，在连续数值特征空间用 KNN 找最近邻（除自身外）；
2. 在样本与其近邻之间按 ``lambda ~ U(0, 1)`` 线性插值；
3. 对连续特征叠加小尺度高斯噪声（``noise_scale * 特征标准差``）以提升多样性；
4. 对 one-hot 分类列不插值，而是按 ``lambda`` 概率从父样本或近邻中随机继承，保持 0/1 取值；
5. 目标值沿用同一 ``lambda`` 插值，保证输入-输出几何关系一致。

默认关闭（``TrainConfig.augment=False``），需要时用 ``--augment`` 开启；测试集始终保留真实数据，
避免评估被合成样本污染。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def augment_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_synthetic: int,
    numeric_columns: list[str],
    sparse_columns: list[str] = (),
    noise_scale: float = 0.03,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """对单目标训练集做 KNN 插值 + 高斯噪声增强。

    Args:
        x_train: 训练特征矩阵（含连续数值列与 ``interface_prep__*`` one-hot 列）。
        y_train: 单目标标签（拉伸或抗剪，已剔除 NaN 的子集）。
        n_synthetic: 目标合成样本条数。
        numeric_columns: 参与 KNN 距离与插值的连续数值列。
        sparse_columns: 稀疏数值列（缺失用哨兵值表示，如 roughness_mm 的 -1），
            不参与插值，按 lambda 从父样本或近邻原样继承（浮点），避免被 int 截断。
        noise_scale: 高斯噪声相对特征标准差的尺度。
        n_neighbors: KNN 候选近邻数。
        random_state: 随机种子，保证可复现。

    Returns:
        (增强后的 X, 增强后的 y)；原始样本在前，合成样本追加在后。
    """
    if n_synthetic <= 0 or len(x_train) < 2:
        return x_train.copy(), y_train.copy()

    rng = np.random.default_rng(random_state)
    numeric = [c for c in numeric_columns if c in x_train.columns]
    sparse = [c for c in sparse_columns if c in x_train.columns]
    onehot = [c for c in x_train.columns if c not in numeric and c not in sparse]

    x_num = x_train[numeric].to_numpy(dtype=float)
    x_sparse = x_train[sparse].to_numpy(dtype=float)
    y_arr = y_train.to_numpy(dtype=float).reshape(-1)

    # 在标准化后的数值特征空间找近邻，避免量纲差异主导距离
    means = x_num.mean(axis=0)
    stds = x_num.std(axis=0)
    stds[stds == 0] = 1.0
    x_scaled = (x_num - means) / stds

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(x_train)), metric="euclidean")
    nn.fit(x_scaled)
    _, indices = nn.kneighbors(x_scaled)
    # 排除自身（每个点最近邻必是自己）
    candidate = indices[:, 1:]

    synth_num = np.empty((n_synthetic, len(numeric)), dtype=float)
    synth_sparse = np.empty((n_synthetic, len(sparse)), dtype=float)
    synth_onehot = np.empty((n_synthetic, len(onehot)), dtype=int)
    synth_y = np.empty(n_synthetic, dtype=float)

    noise_std = noise_scale * stds
    onehot_values = x_train[onehot].to_numpy(dtype=int)

    for i in range(n_synthetic):
        anchor = rng.integers(0, len(x_train))
        neighbor = int(candidate[anchor][rng.integers(0, candidate.shape[1])])
        lam = rng.uniform(0.0, 1.0)

        # 连续特征：线性插值 + 高斯噪声
        interp = x_num[anchor] * (1 - lam) + x_num[neighbor] * lam
        synth_num[i] = interp + rng.normal(0.0, noise_std)

        # one-hot 列：按 lambda 概率从父样本或近邻继承（保持 0/1）
        src = anchor if rng.uniform() > lam else neighbor
        synth_sparse[i] = x_sparse[src]
        synth_onehot[i] = onehot_values[src]

        synth_y[i] = y_arr[anchor] * (1 - lam) + y_arr[neighbor] * lam

    idx = range(len(x_train), len(x_train) + n_synthetic)
    x_synth = pd.concat(
        [
            pd.DataFrame(synth_num, columns=numeric, index=idx),
            pd.DataFrame(synth_sparse, columns=sparse, index=idx),
            pd.DataFrame(synth_onehot, columns=onehot, index=idx),
        ],
        axis=1,
    )
    y_synth = pd.Series(synth_y, index=idx, name=y_train.name)

    x_out = pd.concat([x_train, x_synth], axis=0)
    y_out = pd.concat([y_train, y_synth], axis=0)
    return x_out, y_out


__all__ = ["augment_regression"]
