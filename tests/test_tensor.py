"""对张量自动求导引擎 minitorch.Tensor 的梯度检查测试。

策略：对每个算子，构造一个标量损失 = f(输入)，用 Tensor.backward() 求解析梯度，
再用中心差分（numerical_gradient）求数值梯度，比较相对误差。这是验证 autograd
正确性最可靠的方式，也是整个项目反复使用的手段。
"""
import numpy as np

from minitorch import Tensor, no_grad
from minitorch.utils import numerical_gradient, rel_error


def _check(make_scalar_from_np, x_np, get_analytic_grad, tol=1e-5):
    """通用辅助：比较解析梯度与数值梯度。

    make_scalar_from_np: 纯 numpy 函数 x -> 标量（供数值差分用）
    get_analytic_grad:   x -> 解析梯度（内部用 Tensor.backward() 求）
    """
    num = numerical_gradient(make_scalar_from_np, x_np.copy())
    ana = get_analytic_grad(x_np)
    assert rel_error(ana, num) < tol, f"rel_error={rel_error(ana, num):.2e}"


def test_add_broadcast():
    a_np = np.random.randn(3, 4)
    b_np = np.random.randn(4)        # 触发广播
    a = Tensor(a_np); b = Tensor(b_np)
    (a + b).sum().backward()
    ga = numerical_gradient(lambda x: (x + b_np).sum(), a_np.copy())
    gb = numerical_gradient(lambda x: (a_np + x).sum(), b_np.copy())
    assert rel_error(a.grad, ga) < 1e-5
    assert rel_error(b.grad, gb) < 1e-5


def test_mul_broadcast():
    a_np = np.random.randn(2, 3)
    b_np = np.random.randn(1, 3)
    a = Tensor(a_np); b = Tensor(b_np)
    (a * b).sum().backward()
    ga = numerical_gradient(lambda x: (x * b_np).sum(), a_np.copy())
    gb = numerical_gradient(lambda x: (a_np * x).sum(), b_np.copy())
    assert rel_error(a.grad, ga) < 1e-5
    assert rel_error(b.grad, gb) < 1e-5


def test_matmul():
    A_np = np.random.randn(5, 3)
    B_np = np.random.randn(3, 4)
    A = Tensor(A_np); B = Tensor(B_np)
    (A @ B).sum().backward()
    gA = numerical_gradient(lambda x: (x @ B_np).sum(), A_np.copy())
    gB = numerical_gradient(lambda x: (A_np @ x).sum(), B_np.copy())
    assert rel_error(A.grad, gA) < 1e-5
    assert rel_error(B.grad, gB) < 1e-5


def test_pow_div():
    x_np = np.abs(np.random.randn(4, 4)) + 0.5
    x = Tensor(x_np)
    (x ** 3 / 2.0).sum().backward()
    g = numerical_gradient(lambda v: (v ** 3 / 2.0).sum(), x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_activations():
    x_np = np.random.randn(3, 5)
    for name, np_fn, t_fn in [
        ("relu", lambda v: np.maximum(0, v), lambda t: t.relu()),
        ("sigmoid", lambda v: 1 / (1 + np.exp(-v)), lambda t: t.sigmoid()),
        ("tanh", np.tanh, lambda t: t.tanh()),
        ("exp", np.exp, lambda t: t.exp()),
    ]:
        x = Tensor(x_np)
        t_fn(x).sum().backward()
        g = numerical_gradient(lambda v: np_fn(v).sum(), x_np.copy())
        assert rel_error(x.grad, g) < 1e-5, name


def test_log():
    x_np = np.abs(np.random.randn(3, 3)) + 0.3
    x = Tensor(x_np)
    x.log().sum().backward()
    g = numerical_gradient(lambda v: np.log(v).sum(), x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_sum_mean_axis():
    x_np = np.random.randn(4, 6)
    x = Tensor(x_np)
    x.mean(axis=1).sum().backward()
    g = numerical_gradient(lambda v: v.mean(axis=1).sum(), x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_reshape_transpose_getitem():
    x_np = np.random.randn(2, 6)
    x = Tensor(x_np)
    # reshape -> transpose -> 取一列，再求和
    (x.reshape(3, 4).transpose()[1]).sum().backward()
    g = numerical_gradient(lambda v: v.reshape(3, 4).transpose()[1].sum(), x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_small_mlp_like_loss():
    # 复合：tanh(X W1) @ W2 的均方，相对一个目标
    X = np.random.randn(8, 3)
    W1n = np.random.randn(3, 5)
    W2n = np.random.randn(5, 1)
    Yn = np.random.randn(8, 1)

    def loss_np(W1v):
        H = np.tanh(X @ W1v)
        O = H @ W2n
        return ((O - Yn) ** 2).mean()

    W1 = Tensor(W1n); W2 = Tensor(W2n)
    H = (Tensor(X) @ W1).tanh()
    O = H @ W2
    diff = O - Tensor(Yn)
    (diff * diff).mean().backward()
    g = numerical_gradient(loss_np, W1n.copy())
    assert rel_error(W1.grad, g) < 1e-5


def test_gradient_accumulates_for_reused_tensor():
    x = Tensor(np.array([2.0, 3.0]))
    # L = sum(x*x) + sum(x)  -> dL/dx = 2x + 1
    (((x * x).sum()) + x.sum()).backward()
    assert rel_error(x.grad, 2 * x.data + 1) < 1e-9


def test_no_grad_builds_no_graph():
    x = Tensor(np.random.randn(3))
    with no_grad():
        y = x * 2 + 1
    assert len(y._prev) == 0       # 没有记录前驱
    # backward 不应把梯度传回 x
    y.sum().backward()
    assert np.allclose(x.grad, 0.0)


def test_matches_pytorch():
    torch = __import__("torch")
    Xn = np.random.randn(4, 3)
    Wn = np.random.randn(3, 2)
    X = Tensor(Xn); W = Tensor(Wn)
    ((X @ W).tanh().sum()).backward()

    Xt = torch.tensor(Xn, requires_grad=True)
    Wt = torch.tensor(Wn, requires_grad=True)
    (Xt @ Wt).tanh().sum().backward()

    assert rel_error(X.grad, Xt.grad.numpy()) < 1e-6
    assert rel_error(W.grad, Wt.grad.numpy()) < 1e-6
