"""minitorch —— 从零手写的微型深度学习框架（教学用）。

本包伴随《深度学习手写教学项目》逐里程碑成长，API 刻意贴近 PyTorch，
以便学习者从"亲手实现的原理"平滑过渡到"实际工程里的框架"：

    M2  minitorch.tensor / functional   —— 张量自动求导引擎（框架心脏）
    M3  minitorch.nn                     —— Module / Linear / 激活 / 损失
    M4  minitorch.optim / minitorch.data —— 优化器 / 数据加载
    M5+ nn.conv / nn.rnn / nn.attention  —— 卷积 / 循环 / 注意力等高级层

当前可用（M0 脚手架）：
    minitorch.set_seed        统一随机种子，保证可复现
    minitorch.gradcheck       数值梯度检查（中心差分），全程验证核心
    minitorch.numerical_gradient / rel_error
"""
from minitorch import utils
from minitorch.tensor import Tensor, no_grad, ones, randn, tensor, zeros
from minitorch.utils import gradcheck, numerical_gradient, rel_error, set_seed

__version__ = "0.1.0"

__all__ = [
    "utils",
    "set_seed",
    "gradcheck",
    "numerical_gradient",
    "rel_error",
    # 张量自动求导引擎（框架核心，Part 3 起）
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "no_grad",
    "__version__",
]
