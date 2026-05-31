"""Dropout 正则化。"""
from __future__ import annotations

import numpy as np

from minitorch.nn.module import Module
from minitorch.tensor import Tensor


class Dropout(Module):
    """训练时随机"丢弃"一部分激活，缓解过拟合（推理时不丢弃）。

    采用 **inverted dropout**：训练时把保留的激活除以 ``(1-p)``，这样推理时
    无需做任何缩放，直接原样前向即可。
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * Tensor(mask)        # mask 是常数，梯度只流经被保留的位置
