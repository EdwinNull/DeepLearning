"""数据加载器：负责打乱与分批。"""
from __future__ import annotations

import numpy as np


class DataLoader:
    """按 batch 迭代数据集，可选每轮打乱。

    每次迭代产出 ``(X_batch, y_batch)``（NumPy 数组）。
    """

    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.dataset)
        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, n, self.batch_size):
            batch_idx = idx[i:i + self.batch_size]
            yield self.dataset[batch_idx]          # 整批索引，一次取出

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
