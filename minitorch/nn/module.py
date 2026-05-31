"""nn 的基础设施：Parameter / Module / Sequential（API 贴近 PyTorch）。

- `Parameter`：本质是一个会被自动收集的 `Tensor`，代表"可学习的权重"。
- `Module`：所有层与模型的基类，负责递归收集参数、清零梯度、切换 train/eval。
- `Sequential`：把若干层串起来，依次前向。
"""
from __future__ import annotations

import numpy as np

from minitorch.tensor import Tensor


class Parameter(Tensor):
    """可学习参数：与普通 Tensor 唯一的区别是会被 ``Module.parameters()`` 收集。"""

    pass


class Module:
    """所有网络层/模型的基类。"""

    def __init__(self):
        self.training = True

    # 子类需实现 forward；调用 module(x) 即调用 forward
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """递归收集本模块（及其子模块）中的所有 Parameter。"""
        params = []
        for val in self.__dict__.values():
            if isinstance(val, Parameter):
                params.append(val)
            elif isinstance(val, Module):
                params += val.parameters()
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, Parameter):
                        params.append(v)
                    elif isinstance(v, Module):
                        params += v.parameters()
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def train(self, mode=True):
        """切换到训练模式（影响 Dropout / BatchNorm 等，见 Part 5）。"""
        self.training = mode
        for val in self.__dict__.values():
            if isinstance(val, Module):
                val.train(mode)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, Module):
                        v.train(mode)
        return self

    def eval(self):
        """切换到评估/推理模式。"""
        return self.train(False)


class Sequential(Module):
    """把若干层按顺序串联：前一层的输出是后一层的输入。"""

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
