"""对 minitorch.nn 与 functional 的测试。"""
import numpy as np

import minitorch
from minitorch import Tensor, nn
from minitorch.functional import cross_entropy, log_softmax, softmax
from minitorch.utils import numerical_gradient, rel_error


def test_linear_forward_shapes_and_params():
    lin = nn.Linear(4, 3)
    x = Tensor(np.random.randn(5, 4))
    out = lin(x)
    assert out.shape == (5, 3)
    # weight (4,3) + bias (3,)
    assert len(lin.parameters()) == 2


def test_module_parameters_recursive():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    # 两个 Linear，各 weight+bias = 4 个参数张量
    assert len(model.parameters()) == 4


def test_softmax_sums_to_one():
    x = Tensor(np.random.randn(6, 5))
    p = softmax(x, axis=-1)
    assert np.allclose(p.data.sum(axis=-1), 1.0)


def test_cross_entropy_grad_matches_numerical():
    np.random.seed(0)
    logits_np = np.random.randn(8, 4)
    targets = np.random.randint(0, 4, size=8)

    logits = Tensor(logits_np)
    cross_entropy(logits, targets).backward()

    def loss_np(z):
        z = z - z.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        return -logp[np.arange(8), targets].mean()

    g = numerical_gradient(loss_np, logits_np.copy())
    assert rel_error(logits.grad, g) < 1e-5


def test_cross_entropy_matches_pytorch():
    torch = __import__("torch")
    logits_np = np.random.randn(10, 5)
    targets = np.random.randint(0, 5, size=10)

    logits = Tensor(logits_np)
    loss = cross_entropy(logits, targets)
    loss.backward()

    lt = torch.tensor(logits_np, requires_grad=True)
    lt_loss = torch.nn.functional.cross_entropy(lt, torch.tensor(targets))
    lt_loss.backward()

    assert abs(float(loss.data) - lt_loss.item()) < 1e-6
    assert rel_error(logits.grad, lt.grad.numpy()) < 1e-6


def test_mse_loss():
    pred = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    target = Tensor(np.array([[1.5, 2.0], [2.0, 5.0]]))
    loss = nn.MSELoss()(pred, target)
    expected = np.mean((pred.data - target.data) ** 2)
    assert abs(float(loss.data) - expected) < 1e-9


def test_train_eval_mode_toggle():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    model.eval()
    assert model.training is False and model.layers[0].training is False
    model.train()
    assert model.training is True and model.layers[0].training is True
