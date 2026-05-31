"""全连接层 Linear。"""
from __future__ import annotations

import numpy as np

from minitorch.nn.init import kaiming_uniform
from minitorch.nn.module import Module, Parameter


class Linear(Module):
    """全连接层： y = x @ W + b。

    注意：为教学清晰，这里权重形状是 ``(in_features, out_features)``，前向写作
    ``x @ W``；而 PyTorch 的 ``nn.Linear`` 权重是 ``(out, in)``、前向写作 ``x @ W.T``。
    两者数学等价，只是约定不同（nb 里对照时会提到）。
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(kaiming_uniform((in_features, out_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
