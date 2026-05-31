"""Dropout / BatchNorm / LayerNorm / DataLoader 的测试。"""
import numpy as np

from minitorch import Tensor, data, nn
from minitorch.utils import numerical_gradient, rel_error


def test_dropout_train_vs_eval():
    d = nn.Dropout(0.5)
    x = Tensor(np.ones(2000))
    d.eval()
    assert np.allclose(d(x).data, 1.0)            # eval：原样输出
    d.train()
    out = d(x).data
    frac_zero = (out == 0).mean()
    assert 0.4 < frac_zero < 0.6                   # 约一半被丢弃
    assert np.allclose(out[out > 0], 2.0)          # 保留的被放大 1/(1-p)=2
    assert abs(out.mean() - 1.0) < 0.15            # 期望保持不变


def test_batchnorm_normalizes_and_grad():
    np.random.seed(0)
    x_np = np.random.randn(16, 4) * 3 + 5
    bn = nn.BatchNorm1d(4); bn.train()
    out = bn(Tensor(x_np))
    # gamma=1, beta=0 时，输出应近似零均值单位方差（按特征）
    assert np.allclose(out.data.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(out.data.std(axis=0), 1, atol=1e-2)

    # 梯度检查（训练模式，输出对 x 的梯度）。用随机加权和作损失，
    # 避免归一化输出造成部分梯度分量近零、令相对误差失真。
    R = np.random.randn(16, 4)
    x = Tensor(x_np)
    (bn(x) * Tensor(R)).sum().backward()

    def loss_np(xv):
        mean = xv.mean(0); var = ((xv - mean) ** 2).mean(0)
        xhat = (xv - mean) / np.sqrt(var + bn.eps)
        out = xhat * bn.gamma.data + bn.beta.data
        return (out * R).sum()

    g = numerical_gradient(loss_np, x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_layernorm_grad():
    np.random.seed(1)
    x_np = np.random.randn(6, 5)
    ln = nn.LayerNorm(5)
    R = np.random.randn(6, 5)
    x = Tensor(x_np)
    (ln(x) * Tensor(R)).sum().backward()

    def loss_np(xv):
        mean = xv.mean(-1, keepdims=True); var = ((xv - mean) ** 2).mean(-1, keepdims=True)
        xhat = (xv - mean) / np.sqrt(var + ln.eps)
        out = xhat * ln.gamma.data + ln.beta.data
        return (out * R).sum()

    g = numerical_gradient(loss_np, x_np.copy())
    assert rel_error(x.grad, g) < 1e-5


def test_dataloader_batches_and_coverage():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    ds = data.TensorDataset(X, y)
    dl = data.DataLoader(ds, batch_size=3, shuffle=False)
    assert len(dl) == 4
    sizes = [len(yb) for _, yb in dl]
    assert sizes == [3, 3, 3, 1]
    seen = np.concatenate([yb for _, yb in data.DataLoader(ds, batch_size=3, shuffle=True)])
    assert sorted(seen.tolist()) == list(range(10))   # 打乱后仍覆盖全部
