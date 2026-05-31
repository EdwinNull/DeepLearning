"""随机梯度下降 SGD（支持 momentum / weight_decay / nesterov）。"""
from __future__ import annotations

import numpy as np

from minitorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data          # L2 正则等价于权重衰减
            if self.momentum:
                v = self.velocities[i]
                v[...] = self.momentum * v + g              # 累积动量
                g = g + self.momentum * v if self.nesterov else v
            p.data -= self.lr * g
