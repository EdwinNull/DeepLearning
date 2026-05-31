"""损失函数（包装成 Module）。"""
from __future__ import annotations

from minitorch.functional import cross_entropy
from minitorch.nn.module import Module


class MSELoss(Module):
    """均方误差，用于回归。pred 与 target 均为 Tensor。"""

    def forward(self, pred, target):
        diff = pred - target
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    """多分类交叉熵（内部融合 softmax，数值稳定）。

    logits: (N, C) Tensor；targets: (N,) 整数标签。
    """

    def forward(self, logits, targets):
        return cross_entropy(logits, targets)
