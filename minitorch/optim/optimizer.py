"""优化器基类。

优化器负责"拿到梯度后如何更新参数"。把它抽象出来后，训练循环只需调用
``optimizer.step()`` 和 ``optimizer.zero_grad()``，无需手写参数遍历。
"""
from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        """清零所有参数的梯度（每步训练前调用）。"""
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        raise NotImplementedError
