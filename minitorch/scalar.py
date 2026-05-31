"""标量自动求导引擎（micrograd 风格）—— 自动求导的"最小可理解版"。

这是整个项目"造轮子"的第一步。`Value` 把一个**标量**包装起来，在你做四则
运算、tanh/exp/relu 等操作时**自动记录计算图**；之后只需一句 ``loss.backward()``，
它就能用链式法则把梯度自动反传到每一个输入上——这正是 PyTorch/TensorFlow 最核心的
魔法。

> ⚠️ 教学定位：`Value` 是**标量**引擎，每个数都是一个图节点，真正训练会很慢。它是
> 帮助你"看懂" autograd 的踏脚石，**不属于 minitorch 的主框架路径**。从 Part 3 起，
> 我们会把同样的思想升级到基于 NumPy 的张量引擎 `minitorch.Tensor`，那才是后续 CNN /
> RNN / Transformer 真正依赖的核心。

参考：Andrej Karpathy 的 micrograd。
"""
from __future__ import annotations

import math


class Value:
    """包装一个标量，并自动构建用于反向传播的计算图。

    Attributes:
        data: 标量数值（前向结果）。
        grad: 损失对该节点的梯度，``backward()`` 后被填充。
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        # 内部：本节点如何把梯度传给它的输入（默认叶子节点什么都不做）
        self._backward = lambda: None
        self._prev = set(_children)  # 直接前驱（构成计算图的边）
        self._op = _op               # 产生本节点的运算名（仅用于可视化/调试）

    # ---------------------------------------------------------------- 基本运算
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # 加法：梯度原样分给两个加数（局部导数都是 1）
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # 乘法：对每个因子的局部导数是"另一个因子"
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "只支持常数次幂"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ---------------------------------------------------------------- 激活函数
    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad  # tanh' = 1 - tanh^2

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad  # exp' = exp

        out._backward = _backward
        return out

    # ---------------------------------------------------------------- 反向传播
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
        self.grad = 1.0                  # dL/dL = 1
        for v in reversed(topo):         # 从输出往输入
            v._backward()

    # ---------------------------------------------------- 便捷运算符（基于以上）
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
