"""卷积/池化的梯度检查与 PyTorch 对照。"""
import numpy as np

from minitorch import Tensor, nn
from minitorch.utils import numerical_gradient, rel_error


def test_conv2d_grad_wrt_input():
    np.random.seed(0)
    x_np = np.random.randn(2, 3, 7, 7)
    conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
    R = np.random.randn(2, 4, 7, 7)
    x = Tensor(x_np)
    (conv(x) * Tensor(R)).sum().backward()

    def loss_np(xv):
        return float((conv(Tensor(xv)).data * R).sum())

    g = numerical_gradient(loss_np, x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_conv2d_grad_wrt_weight():
    np.random.seed(1)
    x_np = np.random.randn(2, 3, 6, 6)
    conv = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=0)
    R = np.random.randn(2, 5, 4, 4)
    (conv(Tensor(x_np)) * Tensor(R)).sum().backward()
    w0 = conv.weight.data.copy()

    def loss_np(wv):
        conv.weight.data = wv
        out = float((conv(Tensor(x_np)).data * R).sum())
        conv.weight.data = w0
        return out

    g = numerical_gradient(loss_np, w0.copy())
    assert rel_error(conv.weight.grad, g) < 1e-5


def test_conv2d_matches_pytorch():
    torch = __import__("torch")
    np.random.seed(2)
    x_np = np.random.randn(2, 3, 8, 8)
    conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)

    out = conv(Tensor(x_np)).data

    xt = torch.tensor(x_np)
    wt = torch.tensor(conv.weight.data)
    bt = torch.tensor(conv.bias.data)
    out_t = torch.nn.functional.conv2d(xt, wt, bt, stride=1, padding=1).numpy()
    assert rel_error(out, out_t) < 1e-6


def test_maxpool_grad():
    np.random.seed(3)
    x_np = np.random.randn(2, 3, 8, 8)
    pool = nn.MaxPool2d(2)
    R = np.random.randn(2, 3, 4, 4)
    x = Tensor(x_np)
    (pool(x) * Tensor(R)).sum().backward()

    def loss_np(xv):
        return float((pool(Tensor(xv)).data * R).sum())

    g = numerical_gradient(loss_np, x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_flatten():
    x = Tensor(np.random.randn(5, 3, 4, 4))
    out = nn.Flatten()(x)
    assert out.shape == (5, 48)
