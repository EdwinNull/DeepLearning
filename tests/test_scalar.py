"""对标量自动求导引擎 minitorch.scalar.Value 的测试。

用"有限差分"作为梯度真值来校验 Value.backward() 的正确性——这与 notebook 里
教学用的数值梯度检查思想一致，但这里直接对单个 Value 输入做扰动。
"""
import math

from minitorch.scalar import Value


def _fd(f, x, eps=1e-6):
    """对单变量函数 f 在 x 处做中心差分。f 接收 float、返回 float。"""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_add_mul_pow_grads():
    a = Value(2.0)
    b = Value(-3.0)
    # L = (a*b + a**2)
    L = a * b + a ** 2
    L.backward()
    # dL/da = b + 2a = -3 + 4 = 1 ; dL/db = a = 2
    assert abs(a.grad - 1.0) < 1e-9
    assert abs(b.grad - 2.0) < 1e-9


def test_tanh_grad_matches_finite_difference():
    x0 = 0.7
    x = Value(x0)
    y = x.tanh()
    y.backward()
    expected = _fd(lambda v: math.tanh(v), x0)
    assert abs(x.grad - expected) < 1e-6


def test_relu_grad():
    pos = Value(1.5)
    pos.relu().backward()
    assert abs(pos.grad - 1.0) < 1e-9
    neg = Value(-2.0)
    neg.relu().backward()
    assert abs(neg.grad - 0.0) < 1e-9


def test_division_and_chain():
    a = Value(3.0)
    b = Value(4.0)
    L = (a / b).tanh()
    L.backward()
    # 用有限差分核对 dL/da、dL/db
    da = _fd(lambda v: math.tanh(v / 4.0), 3.0)
    db = _fd(lambda v: math.tanh(3.0 / v), 4.0)
    assert abs(a.grad - da) < 1e-6
    assert abs(b.grad - db) < 1e-6


def test_reused_node_accumulates_gradient():
    # a 被用了两次： L = a*a + a  -> dL/da = 2a + 1
    a = Value(3.0)
    L = a * a + a
    L.backward()
    assert abs(a.grad - (2 * 3.0 + 1)) < 1e-9
