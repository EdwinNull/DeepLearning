"""数据集抽象。

约定：``__getitem__`` 接受**一批索引（数组）**，返回对应的一批数据。这样配合
DataLoader 用整批索引切片，比逐样本取再拼接高效得多（教学友好且不慢）。
"""
from __future__ import annotations

import numpy as np


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    """把特征 X 与标签 y 包成数据集。"""

    def __init__(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
