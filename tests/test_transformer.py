"""注意力 / Embedding / Transformer 的梯度检查与形状测试。"""
import numpy as np

from minitorch import Tensor, nn
from minitorch.nn import scaled_dot_product_attention, causal_mask
from minitorch.utils import numerical_gradient, rel_error


def test_sdpa_shapes_and_softmax():
    q = Tensor(np.random.randn(2, 5, 8))
    k = Tensor(np.random.randn(2, 5, 8))
    v = Tensor(np.random.randn(2, 5, 8))
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 5, 8)
    assert np.allclose(attn.data.sum(-1), 1.0)      # 注意力权重每行和为 1


def test_sdpa_grad():
    np.random.seed(0)
    q_np = np.random.randn(2, 4, 6)
    k_np = np.random.randn(2, 4, 6)
    v_np = np.random.randn(2, 4, 6)
    R = np.random.randn(2, 4, 6)
    q = Tensor(q_np)
    out, _ = scaled_dot_product_attention(q, Tensor(k_np), Tensor(v_np))
    (out * Tensor(R)).sum().backward()

    def loss_np(qv):
        o, _ = scaled_dot_product_attention(Tensor(qv), Tensor(k_np), Tensor(v_np))
        return float((o.data * R).sum())

    g = numerical_gradient(loss_np, q_np.copy())
    assert rel_error(q.grad, g) < 1e-4


def test_causal_mask_blocks_future():
    L = 5
    q = Tensor(np.random.randn(1, L, 4))
    _, attn = scaled_dot_product_attention(q, q, q, mask=causal_mask(L))
    upper = attn.data[0][np.triu_indices(L, k=1)]
    assert np.allclose(upper, 0, atol=1e-6)          # 不能关注未来位置


def test_multihead_attention():
    mha = nn.MultiheadAttention(d_model=16, num_heads=4)
    x = Tensor(np.random.randn(2, 6, 16))
    out = mha(x)
    assert out.shape == (2, 6, 16)
    assert mha.attn_weights.shape == (2, 4, 6, 6)    # (N, heads, L, L)


def test_multihead_grad():
    np.random.seed(1)
    mha = nn.MultiheadAttention(8, 2)
    x_np = np.random.randn(2, 4, 8)
    R = np.random.randn(2, 4, 8)
    x = Tensor(x_np)
    (mha(x) * Tensor(R)).sum().backward()

    def loss_np(xv):
        return float((mha(Tensor(xv)).data * R).sum())

    g = numerical_gradient(loss_np, x_np.copy())
    assert rel_error(x.grad, g) < 1e-4


def test_embedding():
    emb = nn.Embedding(10, 4)
    idx = np.array([[1, 2, 3], [4, 5, 6]])
    out = emb(idx)
    assert out.shape == (2, 3, 4)
    out.sum().backward()
    assert emb.weight.grad[1].sum() != 0            # 被用到的行有梯度


def test_transformer_encoder_layer():
    layer = nn.TransformerEncoderLayer(d_model=16, num_heads=4, d_ff=32)
    x = Tensor(np.random.randn(2, 7, 16))
    out = layer(x)
    assert out.shape == (2, 7, 16)


def test_positional_encoding():
    pe = nn.PositionalEncoding(16)
    x = Tensor(np.zeros((1, 5, 16)))
    out = pe(x)
    assert out.shape == (1, 5, 16)
    assert not np.allclose(out.data, 0)             # 加上了位置信息
