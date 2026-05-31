"""把激活函数包装成 Module，方便放进 Sequential。"""
from __future__ import annotations

from minitorch.functional import softmax
from minitorch.nn.module import Module


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return softmax(x, axis=self.axis)
