"""数值梯度检查（中心差分）—— 贯穿整个项目的核心验证工具。

深度学习里最容易写错的就是反向传播的梯度公式。好在我们有一个与具体实现无关的
"真值"可以对照：**数值梯度**。它直接用函数值的微小变化来估计导数，几乎不可能写错，
因此非常适合用来校验我们手写的"解析梯度"是否正确。

中心差分公式：对标量函数 :math:`f` 关于输入 :math:`x_i` 的偏导数，

    df/dx_i ≈ ( f(x + eps * e_i) - f(x - eps * e_i) ) / (2 * eps)

逐元素地估计出来，再与手写公式得到的解析梯度比较相对误差。整个教学项目中，
我们实现的每一个算子/层都会用它来"验收"。
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """用中心差分计算标量函数 ``f`` 在 ``x`` 处对每个元素的数值梯度。

    Args:
        f: 接受 ndarray、返回标量的函数。注意 ``f`` 内部不要原地修改入参。
        x: 求导点；过程中会被临时改动并还原，函数返回时 ``x`` 数值不变。
        eps: 差分步长，过大不准、过小受浮点误差影响，1e-6 通常合适。

    Returns:
        与 ``x`` 同形状的数值梯度数组。
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = float(f(x))
        x[idx] = orig - eps
        f_minus = float(f(x))
        x[idx] = orig  # 还原，避免污染后续计算
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """相对误差 ``max |a-b| / (|a|+|b|)``，对数值量级不敏感，常用于梯度比较。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(eps, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


def gradcheck(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    analytic_grad: np.ndarray,
    eps: float = 1e-6,
    tol: float = 1e-5,
    name: str = "",
    verbose: bool = True,
) -> bool:
    """比较解析梯度与数值梯度，打印 PASS/FAIL，并返回是否通过。

    Args:
        f: 标量函数 ``ndarray -> float``。
        x: 求导点。
        analytic_grad: 手写公式算出的解析梯度（需与 ``x`` 同形状）。
        eps: 差分步长。
        tol: 相对误差阈值，低于它判 PASS。
        name: 检查项名称，便于在输出中区分。
        verbose: 是否打印结果。

    Returns:
        是否通过（``rel_error < tol``）。
    """
    num = numerical_gradient(f, np.array(x, dtype=np.float64), eps=eps)
    err = rel_error(np.asarray(analytic_grad), num)
    passed = bool(err < tol)
    if verbose:
        tag = "PASS ✅" if passed else "FAIL ❌"
        label = f" [{name}]" if name else ""
        print(f"gradcheck{label}: rel_error = {err:.3e}  (tol={tol:.0e})  ->  {tag}")
    return passed
