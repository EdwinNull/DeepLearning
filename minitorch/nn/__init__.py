"""minitorch.nn —— 神经网络的层与损失（API 贴近 PyTorch）。"""
from minitorch.nn import init
from minitorch.nn.activation import ReLU, Sigmoid, Softmax, Tanh
from minitorch.nn.attention import MultiheadAttention, causal_mask, scaled_dot_product_attention
from minitorch.nn.conv import Conv2d, Flatten, MaxPool2d
from minitorch.nn.embedding import Embedding
from minitorch.nn.dropout import Dropout
from minitorch.nn.linear import Linear
from minitorch.nn.loss import CrossEntropyLoss, MSELoss
from minitorch.nn.module import Module, Parameter, Sequential
from minitorch.nn.norm import BatchNorm1d, LayerNorm
from minitorch.nn.rnn import LSTM, RNN, LSTMCell, RNNCell
from minitorch.nn.transformer import PositionalEncoding, TransformerEncoderLayer

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
    "Conv2d",
    "MaxPool2d",
    "Flatten",
    "RNNCell",
    "RNN",
    "LSTMCell",
    "LSTM",
    "Embedding",
    "scaled_dot_product_attention",
    "causal_mask",
    "MultiheadAttention",
    "PositionalEncoding",
    "TransformerEncoderLayer",
]
