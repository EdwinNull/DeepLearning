"""函数式算子：在 Tensor 之上构建的常用函数（softmax / cross_entropy 等）。

这些函数不持有参数，纯粹是对 Tensor 的变换；nn 里的"层"会调用它们。
"""
from __future__ import annotations

import numpy as np

from minitorch.tensor import Tensor


def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def softmax(x, axis=-1):
    """数值稳定的 softmax：先减去最大值（当作常数）再 exp，避免上溢。"""
    m = x.data.max(axis=axis, keepdims=True)     # 常数，不参与求导
    e = (x - Tensor(m)).exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """log(softmax(x))，用 log-sum-exp 技巧保证数值稳定。"""
    m = x.data.max(axis=axis, keepdims=True)
    shifted = x - Tensor(m)
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def cross_entropy(logits, targets):
    """多分类交叉熵（内部融合了 softmax，数值稳定）。

    Args:
        logits: 形状 (N, C) 的 Tensor（未归一化分数）。
        targets: 形状 (N,) 的整数类别标签（numpy 数组或列表）。
    Returns:
        标量 Tensor —— 平均交叉熵损失。
    """
    targets = np.asarray(targets).astype(int)
    n = logits.shape[0]
    logp = log_softmax(logits, axis=-1)            # (N, C)
    picked = logp[(np.arange(n), targets)]         # 取每行正确类别的 log 概率 -> (N,)
    return -picked.mean()
