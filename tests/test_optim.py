"""优化器测试：在一个简单凸问题上，各优化器都应收敛到最优点。"""
import numpy as np

from minitorch import Tensor
from minitorch.optim import SGD, RMSProp, Adam


def _converges(OptClass, steps=1500, **kw):
    target = np.array([1.0, -2.0, 3.0])
    w = Tensor(np.zeros(3))
    opt = OptClass([w], **kw)
    for _ in range(steps):
        opt.zero_grad()
        d = w - Tensor(target)
        (d * d).sum().backward()
        opt.step()
    return np.allclose(w.data, target, atol=1e-2)


def test_sgd():
    assert _converges(SGD, lr=0.1)


def test_sgd_momentum():
    assert _converges(SGD, lr=0.05, momentum=0.9)


def test_rmsprop():
    assert _converges(RMSProp, lr=0.05)


def test_adam():
    assert _converges(Adam, lr=0.1)


def test_zero_grad():
    w = Tensor(np.ones(3))
    opt = SGD([w], lr=0.1)
    (w * w).sum().backward()
    assert not np.allclose(w.grad, 0)
    opt.zero_grad()
    assert np.allclose(w.grad, 0)
