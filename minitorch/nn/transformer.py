"""Transformer 组件：位置编码与 Transformer 编码层。

我们采用 **pre-norm** 结构（LayerNorm 放在子层之前），它比原始 post-norm 更易训练。
每个编码层 = 多头自注意力 + 前馈网络，各自带残差连接。
"""
from __future__ import annotations

import numpy as np

from minitorch.nn.attention import MultiheadAttention
from minitorch.nn.dropout import Dropout
from minitorch.nn.linear import Linear
from minitorch.nn.module import Module
from minitorch.nn.norm import LayerNorm
from minitorch.tensor import Tensor


class PositionalEncoding(Module):
    """正弦位置编码：给每个位置一个固定的向量，让模型感知"顺序"。"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pe = pe

    def forward(self, x):
        # x: (N, L, d_model)
        return x + Tensor(self.pe[:x.shape[1]])


class TransformerEncoderLayer(Module):
    """一个 Transformer 编码层（pre-norm）：

        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
        self.drop = Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))           # 自注意力子层
        h = self.ff2(self.ff1(self.norm2(x)).relu())                # 前馈子层
        x = x + self.drop(h)
        return x
