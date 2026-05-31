"""张量自动求导引擎 —— minitorch 的"心脏"。

Part 2 的 `Value` 把每个**标量**当作图节点，正确但太慢。这里我们把同样的 autograd
思想升级到**张量**：一个 `Tensor` 节点包住一整个 NumPy 数组，用向量化一次算完一层。
后续的 nn.Module、优化器、CNN / RNN / Transformer 全部建立在它之上。

设计要点：
- 每个运算正向算出 `data`，并挂一个 `_backward` 闭包，知道如何把输出梯度分配给输入；
- `backward()` 对计算图做拓扑排序，再逆序调用各节点的 `_backward`；
- **广播 (broadcasting)** 的反向用 `_unbroadcast` 把梯度"求和还原"回原形状；
- 为教学清晰，内部统一使用 float64（数值梯度检查更精确）。
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

# 全局开关：是否构建计算图（no_grad() 下关闭，用于推理/验证，省内存且更快）
_GRAD_ENABLED = True


@contextmanager
def no_grad():
    """上下文管理器：其内部的运算不记录计算图（类似 ``torch.no_grad()``）。"""
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def _unbroadcast(grad, shape):
    """把 ``grad`` 沿"被广播出来的维度"求和，还原成 ``shape``（广播的反向操作）。"""
    # 1) 去掉前面多出来的维度
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) 对原本是 1、被广播放大的维度求和
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """包住一个 NumPy 数组，并自动构建用于反向传播的计算图。"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype=np.float64)
        if self.data.dtype != np.float64:
            self.data = self.data.astype(np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children) if _GRAD_ENABLED else set()
        self._op = _op

    # ----------------------------------------------------------------- 工具
    def _ensure(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    # ------------------------------------------------------------- 逐元素运算
    def __add__(self, other):
        other = self._ensure(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._ensure(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "只支持常数次幂"
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += _unbroadcast(p * self.data ** (p - 1) * out.grad, self.data.shape)

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    # --------------------------------------------------------------- 矩阵乘法
    def matmul(self, other):
        other = self._ensure(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            ga = out.grad @ np.swapaxes(other.data, -1, -2)   # dA = dC · B^T
            gb = np.swapaxes(self.data, -1, -2) @ out.grad     # dB = A^T · dC
            self.grad += _unbroadcast(ga, self.data.shape)
            other.grad += _unbroadcast(gb, other.data.shape)

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    __matmul__ = matmul

    # ------------------------------------------------------------------- 归约
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis if isinstance(axis, int) else tuple(axis))
            self.grad += np.broadcast_to(grad, self.data.shape)

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.data.shape[a] for a in axes]))

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis if isinstance(axis, int) else tuple(axis))
            self.grad += np.broadcast_to(grad, self.data.shape) / count

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    # ------------------------------------------------------------- 激活/逐元素
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    # ----------------------------------------------------------------- 形状变换
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(np.transpose(self.data, axes), (self,), "transpose")

        def _backward():
            if axes is None:
                self.grad += np.transpose(out.grad)
            else:
                inv = np.argsort(axes)
                self.grad += np.transpose(out.grad, inv)

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)   # 把梯度散射回被索引的位置（支持重复索引）
            self.grad += g

        if _GRAD_ENABLED:
            out._backward = _backward
        return out

    # ------------------------------------------------------------------- 反向
    def backward(self):
        """从本节点出发，按拓扑序反向传播，填充图中所有节点的 ``grad``。"""
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)   # dL/dL = 1
        for v in reversed(topo):
            v._backward()

    def detach(self):
        """返回一个共享数值、但脱离计算图的新张量（不追踪梯度）。"""
        return Tensor(self.data.copy())

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # ----------------------------------------------- 便捷运算符（基于上面的实现）
    def __neg__(self):
        return self * -1

    def __radd__(self, o):
        return self + o

    def __sub__(self, o):
        return self + (-self._ensure(o))

    def __rsub__(self, o):
        return self._ensure(o) + (-self)

    def __rmul__(self, o):
        return self * o

    def __truediv__(self, o):
        return self * (self._ensure(o) ** -1)

    def __rtruediv__(self, o):
        return self._ensure(o) * (self ** -1)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"


# ----------------------------------------------------------------- 工厂函数
def tensor(data):
    return Tensor(data)


def zeros(*shape):
    return Tensor(np.zeros(shape))


def ones(*shape):
    return Tensor(np.ones(shape))


def randn(*shape):
    return Tensor(np.random.randn(*shape))
