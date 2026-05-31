"""注意力机制：scaled dot-product attention 与 MultiheadAttention。

注意力让模型"一步到位"地关注序列中任意位置（不像 RNN 必须顺序传递），是 Transformer
与大模型时代的基石。这里全部用我们自己的 Tensor 算子搭建，反向由 autograd 自动完成。
"""
from __future__ import annotations

import numpy as np

from minitorch.functional import softmax
from minitorch.nn.linear import Linear
from minitorch.nn.module import Module
from minitorch.tensor import Tensor


def scaled_dot_product_attention(q, k, v, mask=None):
    """缩放点积注意力。

    Args:
        q: (B, Lq, d)   查询
        k: (B, Lk, d)   键
        v: (B, Lk, dv)  值
        mask: 可加性掩码，(Lq, Lk) 或可广播到 (B, Lq, Lk)，被屏蔽处取一个很大的负数。
    Returns:
        (输出 (B, Lq, dv), 注意力权重 (B, Lq, Lk))
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(0, 2, 1)) * (1.0 / np.sqrt(d))   # (B, Lq, Lk)
    if mask is not None:
        scores = scores + Tensor(mask)
    attn = softmax(scores, axis=-1)
    return attn @ v, attn


def causal_mask(L):
    """因果掩码：位置 i 只能看到 <= i 的位置（用于自回归）。"""
    m = np.triu(np.ones((L, L)), k=1) * (-1e9)
    return m


class MultiheadAttention(Module):
    """多头自注意力。把 d_model 拆成 num_heads 个子空间并行做注意力再拼接。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.dk = d_model // num_heads
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        self.attn_weights = None

    def _split(self, x, N, L):
        # (N, L, d_model) -> (N*h, L, dk)
        return x.reshape(N, L, self.h, self.dk).transpose(0, 2, 1, 3).reshape(N * self.h, L, self.dk)

    def _merge(self, x, N, L):
        # (N*h, L, dk) -> (N, L, d_model)
        return x.reshape(N, self.h, L, self.dk).transpose(0, 2, 1, 3).reshape(N, L, self.d_model)

    def forward(self, x, mask=None):
        N, L, _ = x.shape
        q = self._split(self.Wq(x), N, L)
        k = self._split(self.Wk(x), N, L)
        v = self._split(self.Wv(x), N, L)
        out, attn = scaled_dot_product_attention(q, k, v, mask)   # (N*h, L, dk)
        self.attn_weights = attn.data.reshape(N, self.h, L, L)    # 存下来便于可视化
        return self.Wo(self._merge(out, N, L))
