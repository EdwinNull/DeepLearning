"""数据集加载工具（目前提供 MNIST）。

策略：用 torchvision 下载 MNIST 原始文件到本地 ``data/`` 并缓存（只下载一次），
再转成 NumPy 数组返回，供 minitorch 训练使用。torchvision 仅在此用作"下载器 +
解析器"，与训练本身无关。
"""
from __future__ import annotations

import numpy as np


def load_mnist(root="data", n_train=None, n_test=None, flatten=True):
    """加载 MNIST，返回 ``(X_train, y_train), (X_test, y_test)``。

    X 为 float64，像素已归一化到 [0, 1]；y 为 int 标签 (0-9)。

    Args:
        root: 数据缓存目录。
        n_train / n_test: 只取前若干个样本（教学/提速用），None 表示全部。
        flatten: 是否把 28x28 展平成 784 维向量。
    """
    from torchvision import datasets  # 延迟导入：只有真正用到才需要 torchvision

    train = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)

    x_tr = train.data.numpy().astype(np.float64) / 255.0
    y_tr = train.targets.numpy().astype(int)
    x_te = test.data.numpy().astype(np.float64) / 255.0
    y_te = test.targets.numpy().astype(int)

    if flatten:
        x_tr = x_tr.reshape(len(x_tr), -1)
        x_te = x_te.reshape(len(x_te), -1)

    if n_train is not None:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    if n_test is not None:
        x_te, y_te = x_te[:n_test], y_te[:n_test]

    return (x_tr, y_tr), (x_te, y_te)
