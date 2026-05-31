"""minitorch.nn —— 神经网络的层与损失（API 贴近 PyTorch）。"""
from minitorch.nn import init
from minitorch.nn.activation import ReLU, Sigmoid, Softmax, Tanh
from minitorch.nn.dropout import Dropout
from minitorch.nn.linear import Linear
from minitorch.nn.loss import CrossEntropyLoss, MSELoss
from minitorch.nn.module import Module, Parameter, Sequential
from minitorch.nn.norm import BatchNorm1d, LayerNorm

__all__ = [
    "init",
    "Module",
    "Parameter",
    "Sequential",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "MSELoss",
    "CrossEntropyLoss",
    "Dropout",
    "BatchNorm1d",
    "LayerNorm",
]
