"""权重初始化策略。

初始化看似不起眼，却直接影响网络能否顺利训练（梯度消失/爆炸往往源于糟糕的初始化）。
这里实现两类最常用的方法，nb10 会详细讨论它们的来由。
"""
from __future__ import annotations

import numpy as np


def xavier_uniform(shape):
    """Xavier/Glorot 均匀初始化：适合 tanh/sigmoid 等对称激活。"""
    fan_in, fan_out = shape[-2], shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def kaiming_normal(shape):
    """Kaiming/He 正态初始化：适合 ReLU 系列激活。"""
    fan_in = shape[-2]
    return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)


def kaiming_uniform(shape):
    """Kaiming/He 均匀初始化（minitorch 中 Linear 的默认）。"""
    fan_in = shape[-2]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)
