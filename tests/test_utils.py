"""对 minitorch.utils 的测试：数值梯度检查与随机种子。

这是整个项目 ``tests/`` 的第一块，后续每个里程碑都会往这里追加对应算子/层的
梯度检查测试，保证 notebook 中沉淀进包的实现长期正确、可重复运行。
"""
import numpy as np

import minitorch
from minitorch import gradcheck, numerical_gradient, rel_error, set_seed


def test_numerical_gradient_quadratic():
    # f(x) = sum(x^2)  =>  grad = 2x
    set_seed(0)
    x = np.random.randn(5)
    num = numerical_gradient(lambda v: np.sum(v**2), x)
    assert rel_error(num, 2 * x) < 1e-5


def test_numerical_gradient_does_not_mutate_input():
    x = np.array([1.0, -2.0, 3.0])
    x_copy = x.copy()
    numerical_gradient(lambda v: np.sum(v**3), x)
    assert np.array_equal(x, x_copy)  # 求导后入参应保持不变


def test_gradcheck_pass_and_fail():
    x = np.random.randn(4)
    # 正确的解析梯度 -> 应通过
    assert gradcheck(lambda v: np.sum(v**2), x, 2 * x, name="correct", verbose=False)
    # 故意错误的解析梯度 -> 应不通过
    assert not gradcheck(lambda v: np.sum(v**2), x, 3 * x, name="wrong", verbose=False)


def test_set_seed_reproducible():
    set_seed(123)
    a = np.random.randn(10)
    set_seed(123)
    b = np.random.randn(10)
    assert np.array_equal(a, b)


def test_package_metadata():
    assert isinstance(minitorch.__version__, str)
    assert hasattr(minitorch, "set_seed")
