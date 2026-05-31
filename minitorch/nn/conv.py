"""卷积相关层：Conv2d、MaxPool2d、Flatten。

核心思路（im2col）：把"卷积"转化为"矩阵乘法"——
1. `unfold`(im2col) 把每个卷积窗口展平成一列，得到 (N, C*kh*kw, L)；
2. 把卷积核 reshape 成 (out_ch, C*kh*kw)，与上面做矩阵乘法即得卷积结果。

这样卷积就复用了我们已经验证过的 `matmul` autograd，唯一需要自定义反向的只有
`unfold` 一步（它的反向是 col2im：把梯度散射回原图对应位置）。
"""
from __future__ import annotations

import numpy as np

from minitorch.nn.init import kaiming_uniform
from minitorch.nn.module import Module, Parameter
from minitorch.tensor import Tensor, is_grad_enabled


# ----------------------------------------------------------------- im2col / col2im
def _im2col(x, kh, kw, stride, pad):
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.zeros((N, C, kh, kw, out_h, out_w))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = xpad[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride]
    return cols.reshape(N, C * kh * kw, out_h * out_w), out_h, out_w


def _col2im(grad_cols, x_shape, kh, kw, stride, pad, out_h, out_w):
    N, C, H, W = x_shape
    cols = grad_cols.reshape(N, C, kh, kw, out_h, out_w)
    xpad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for i in range(kh):
        for j in range(kw):
            xpad[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] += cols[:, :, i, j, :, :]
    return xpad[:, :, pad:pad + H, pad:pad + W]


def unfold(x, kh, kw, stride=1, pad=0):
    """im2col：返回 (N, C*kh*kw, out_h*out_w) 的 Tensor，反向用 col2im。"""
    cols_np, oh, ow = _im2col(x.data, kh, kw, stride, pad)
    out = Tensor(cols_np, (x,), "unfold")

    def _backward():
        x.grad += _col2im(out.grad, x.data.shape, kh, kw, stride, pad, oh, ow)

    if is_grad_enabled():
        out._backward = _backward
    return out, oh, ow


# ----------------------------------------------------------------- Conv2d
class Conv2d(Module):
    """二维卷积。权重形状 (out_ch, in_ch, kh, kw)，与 PyTorch 一致。"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        kh = kw = kernel_size
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kh, self.kw = kh, kw
        self.stride, self.padding = stride, padding
        # 用 kaiming，fan_in = in_ch*kh*kw
        w = kaiming_uniform((in_ch * kh * kw, out_ch)).reshape(in_ch, kh, kw, out_ch)
        self.weight = Parameter(w.transpose(3, 0, 1, 2))            # (out_ch, in_ch, kh, kw)
        self.bias = Parameter(np.zeros(out_ch)) if bias else None

    def forward(self, x):
        N = x.shape[0]
        cols, oh, ow = unfold(x, self.kh, self.kw, self.stride, self.padding)  # (N, in*kh*kw, L)
        w2 = self.weight.reshape(self.out_ch, self.in_ch * self.kh * self.kw)  # (out_ch, in*kh*kw)
        out = w2 @ cols                                                        # (N, out_ch, L)
        if self.bias is not None:
            out = out + self.bias.reshape(1, self.out_ch, 1)
        return out.reshape(N, self.out_ch, oh, ow)


# ----------------------------------------------------------------- MaxPool2d
class MaxPool2d(Module):
    """最大池化（非重叠，stride 默认等于 kernel_size）。"""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.k = kernel_size
        self.stride = stride or kernel_size
        assert self.stride == self.k, "本教学实现仅支持非重叠池化(stride==kernel)"

    def forward(self, x):
        k = self.k
        N, C, H, W = x.shape
        oh, ow = H // k, W // k
        xr = x.data.reshape(N, C, oh, k, ow, k).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, oh, ow, k * k)
        arg = xr.argmax(axis=-1)
        out = Tensor(xr.max(axis=-1), (x,), "maxpool")

        def _backward():
            dxr = np.zeros((N, C, oh, ow, k * k))
            ii, jj, pp, qq = np.indices((N, C, oh, ow))
            dxr[ii, jj, pp, qq, arg] = out.grad
            x.grad += dxr.reshape(N, C, oh, ow, k, k).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)

        if is_grad_enabled():
            out._backward = _backward
        return out


# ----------------------------------------------------------------- Flatten
class Flatten(Module):
    """把 (N, C, H, W) 展平成 (N, C*H*W)，便于接全连接层。"""

    def forward(self, x):
        return x.reshape(x.shape[0], -1)
