"""循环神经网络：RNNCell / RNN / LSTMCell / LSTM。

教学要点：序列模型在时间上"展开"成一张很深的计算图；它的反向传播叫
**BPTT（随时间反向传播）**。而我们**完全不必手写 BPTT**——只要把每个时间步的前向
用 Tensor 算子写出来，autograd 会自动沿展开的图反传。这正是自动求导引擎的威力。
"""
from __future__ import annotations

import numpy as np

from minitorch.nn.init import kaiming_uniform
from minitorch.nn.module import Module, Parameter
from minitorch.tensor import Tensor


class RNNCell(Module):
    """单步 vanilla RNN： h' = tanh(x Wxh + h Whh + b)。"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wxh = Parameter(kaiming_uniform((input_size, hidden_size)))
        self.Whh = Parameter(kaiming_uniform((hidden_size, hidden_size)))
        self.bh = Parameter(np.zeros(hidden_size))

    def forward(self, x, h):
        return (x @ self.Wxh + h @ self.Whh + self.bh).tanh()


class RNN(Module):
    """在整个序列上展开 RNNCell。输入 x: (N, T, input_size)。

    返回 (outputs, h_n)：outputs 是长度 T 的列表，每个 (N, hidden)；h_n 是最后隐藏态。
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        N, T, _ = x.shape
        h = h0 if h0 is not None else Tensor(np.zeros((N, self.hidden_size)))
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)      # 时间步展开 -> autograd 自动 BPTT
            outputs.append(h)
        return outputs, h


class LSTMCell(Module):
    """单步 LSTM（输入/遗忘/输出门 + 候选记忆）。

    四个门用一个大矩阵一次算出，再切分，效率更高。
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        H = hidden_size
        self.H = H
        self.Wx = Parameter(kaiming_uniform((input_size, 4 * H)))
        self.Wh = Parameter(kaiming_uniform((H, 4 * H)))
        b = np.zeros(4 * H)
        b[H:2 * H] = 1.0          # 遗忘门偏置初始化为 1：让 LSTM 默认"记住"，利于学习长依赖
        self.b = Parameter(b)

    def forward(self, x, state):
        h, c = state
        H = self.H
        z = x @ self.Wx + h @ self.Wh + self.b      # (N, 4H)
        i = z[:, 0:H].sigmoid()                      # 输入门
        f = z[:, H:2 * H].sigmoid()                  # 遗忘门
        g = z[:, 2 * H:3 * H].tanh()                 # 候选记忆
        o = z[:, 3 * H:4 * H].sigmoid()              # 输出门
        c_new = f * c + i * g                        # 更新记忆
        h_new = o * c_new.tanh()                     # 输出隐藏态
        return h_new, c_new


class LSTM(Module):
    """在整个序列上展开 LSTMCell。输入 x: (N, T, input_size)。"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x, state=None):
        N, T, _ = x.shape
        if state is None:
            h = Tensor(np.zeros((N, self.hidden_size)))
            c = Tensor(np.zeros((N, self.hidden_size)))
        else:
            h, c = state
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h)
        return outputs, (h, c)
