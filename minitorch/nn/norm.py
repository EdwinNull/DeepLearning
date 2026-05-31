"""归一化层：BatchNorm1d 与 LayerNorm。

教学要点：归一化的反向公式相当复杂（尤其 BatchNorm），但**我们不必手推**——只要
用 Tensor 算子把前向写出来，autograd 就会自动、正确地反传。nb14 会带你手推闭式
反向公式来加深理解，并用 gradcheck 证明它与 autograd 一致。
"""
from __future__ import annotations

import numpy as np

from minitorch.nn.module import Module, Parameter
from minitorch.tensor import Tensor


class BatchNorm1d(Module):
    """对一个 batch 在**特征维**上做归一化。

    训练时用当前 batch 的均值/方差归一化，并维护"运行统计量"；推理时改用运行统计量。
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = Parameter(np.ones(num_features))    # 可学习的缩放
        self.beta = Parameter(np.zeros(num_features))    # 可学习的平移
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)                         # (features,)
            var = ((x - mean) * (x - mean)).mean(axis=0)
            # 更新运行统计量（用 .data，不参与求导）
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            x_hat = (x - mean) / ((var + self.eps) ** 0.5)
        else:
            x_hat = (x - Tensor(self.running_mean)) / ((Tensor(self.running_var) + self.eps) ** 0.5)
        return x_hat * self.gamma + self.beta


class LayerNorm(Module):
    """对每个样本在**最后一维**上做归一化（Transformer 的标配）。"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = Parameter(np.ones(normalized_shape))
        self.beta = Parameter(np.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) * (x - mean)).mean(axis=-1, keepdims=True)
        x_hat = (x - mean) / ((var + self.eps) ** 0.5)
        return x_hat * self.gamma + self.beta
