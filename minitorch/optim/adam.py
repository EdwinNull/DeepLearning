"""自适应学习率优化器：RMSProp 与 Adam。"""
from __future__ import annotations

import numpy as np

from minitorch.optim.optimizer import Optimizer


class RMSProp(Optimizer):
    """用梯度平方的滑动平均自适应缩放每个参数的步长。"""

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.sq = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad
            self.sq[i] = self.alpha * self.sq[i] + (1 - self.alpha) * g * g
            p.data -= self.lr * g / (np.sqrt(self.sq[i]) + self.eps)


class Adam(Optimizer):
    """Adam = Momentum + RMSProp + 偏差校正。深度学习最常用的优化器。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]   # 一阶矩（动量）
        self.v = [np.zeros_like(p.data) for p in self.params]   # 二阶矩（梯度平方）
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)         # 偏差校正
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
