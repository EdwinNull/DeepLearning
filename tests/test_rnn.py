"""RNN / LSTM 的梯度检查（验证 BPTT 由 autograd 自动且正确地完成）。"""
import numpy as np

from minitorch import Tensor, nn
from minitorch.utils import numerical_gradient, rel_error


def test_rnncell_grad():
    np.random.seed(0)
    cell = nn.RNNCell(3, 4)
    x_np = np.random.randn(2, 3)
    h_np = np.random.randn(2, 4)
    R = np.random.randn(2, 4)
    x = Tensor(x_np)
    (cell(x, Tensor(h_np)) * Tensor(R)).sum().backward()

    def loss_np(xv):
        return float((cell(Tensor(xv), Tensor(h_np)).data * R).sum())

    g = numerical_gradient(loss_np, x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_rnn_bptt_grad_over_sequence():
    # 验证整条序列展开后，对输入的梯度（即 BPTT）正确
    np.random.seed(1)
    rnn = nn.RNN(2, 5)
    x_np = np.random.randn(3, 6, 2)        # (N, T, input)
    R = np.random.randn(3, 5)
    x = Tensor(x_np)
    _, h_n = rnn(x)
    (h_n * Tensor(R)).sum().backward()

    def loss_np(xv):
        _, hn = rnn(Tensor(xv))
        return float((hn.data * R).sum())

    g = numerical_gradient(loss_np, x_np.copy())
    assert rel_error(x.grad, g) < 1e-4


def test_lstmcell_grad():
    np.random.seed(2)
    cell = nn.LSTMCell(3, 4)
    x_np = np.random.randn(2, 3)
    h_np = np.random.randn(2, 4)
    c_np = np.random.randn(2, 4)
    R = np.random.randn(2, 4)
    x = Tensor(x_np)
    h_new, _ = cell(x, (Tensor(h_np), Tensor(c_np)))
    (h_new * Tensor(R)).sum().backward()

    def loss_np(xv):
        hn, _ = cell(Tensor(xv), (Tensor(h_np), Tensor(c_np)))
        return float((hn.data * R).sum())

    g = numerical_gradient(loss_np, x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_lstm_runs_over_sequence():
    lstm = nn.LSTM(4, 8)
    x = Tensor(np.random.randn(5, 7, 4))
    outputs, (h, c) = lstm(x)
    assert len(outputs) == 7
    assert h.shape == (5, 8) and c.shape == (5, 8)
