"""词嵌入 Embedding：把整数 token id 映射成可学习的向量。"""
from __future__ import annotations

import numpy as np

from minitorch.nn.module import Module, Parameter


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.1)

    def forward(self, idx):
        """idx: 整数数组 (...)，返回查表结果 (..., embedding_dim)。"""
        return self.weight[np.asarray(idx).astype(int)]
