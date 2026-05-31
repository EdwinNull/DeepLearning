# 深度学习手写教学项目 · minitorch

> 面向「有 Python 基础、深度学习零基础」的学习者：用 Jupyter notebook **从零亲手实现**深度学习的核心函数与框架。我们会一路从最简单的梯度下降出发，亲手造出一个 API 贴近 PyTorch 的微型框架 **minitorch**，并最终用它训练 CNN、RNN/LSTM 与 Transformer。

## 这个项目是什么

市面上大多数教程教你**调用** PyTorch / TensorFlow 的 API；本项目教你**亲手把这些 API 造出来**。我们坚信：只有自己实现过一遍反向传播、卷积、注意力，才能真正理解深度学习"为什么有效"。

整个项目围绕一条"造轮子"主线展开：

```
手推梯度  →  标量自动求导引擎  →  张量自动求导(autograd)  →
nn.Module / 层  →  优化器 / 数据加载  →  CNN  →  RNN/LSTM  →  Transformer
```

每个核心知识点都遵循同一套教学法：

1. **直觉 + 数学**：先讲清"为什么"，再给必要的公式；
2. **从零手写**：只用 NumPy 把它实现出来；
3. **数值验证**：用数值梯度检查（中心差分）确认我们的梯度公式正确；
4. **PyTorch 对照**：和官方实现比对，确认结果一致，并为将来的工程实践打基础。

我们手写的代码会逐步沉淀进可复用的 `minitorch` 包——后续 notebook 直接 `import`，不重复造轮子。

## 学习路线图

| Part | 主题 | 你将亲手实现 |
| :--: | --- | --- |
| 0 | 引导与环境 | 环境自检、全景路线图 |
| 1 | 数学与梯度直觉 | 线性回归、梯度下降、手推反向传播 |
| 2 | 标量自动求导 | `Value` 引擎（micrograd 风格） |
| 3 | 张量自动求导 | `Tensor` + 广播 / matmul 反向（**框架心脏**） |
| 4 | 神经网络抽象 | `Module` / `Linear` / 激活 / 损失，训练 MLP 跑 MNIST |
| 5 | 优化与训练工程 | SGD / Adam、DataLoader、Dropout、BatchNorm / LayerNorm |
| 6 | 卷积神经网络 | `Conv2d`（im2col）、池化，训练 CNN |
| 7 | 循环神经网络 | `RNNCell` / `LSTMCell`、BPTT |
| 8 | 注意力与 Transformer | 注意力、多头、位置编码，训练 mini-Transformer |

> 课程共 8 个 Part、23 个核心 notebook，外加引导与收尾。建议按编号顺序学习。

## 快速开始

需要 Python ≥ 3.10（推荐 3.11）。

```bash
# 1) 安装运行时依赖
pip install -r requirements.txt

# torch 仅用于「对照」，建议安装体积更小的 CPU 版：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 若该专用源在你的网络下不可用，直接 pip install torch torchvision 即可（wheel 较大但同样能用）。

# 2) 以「可编辑模式」安装 minitorch，让所有 notebook 都能 import 它
pip install -e .

# 3) 启动 Jupyter，从 notebooks/part0_setup/00_setup_and_overview.ipynb 开始
jupyter lab
```

### （可选）运行测试

`minitorch` 的每个核心算子 / 层都配有数值梯度检查测试：

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

## 目录结构

```
.
├── minitorch/          # 我们亲手造的微型深度学习框架
│   ├── tensor.py       #   张量自动求导引擎（框架心脏）
│   ├── functional.py   #   softmax / cross_entropy 等函数式算子
│   ├── scalar.py       #   标量 autograd 引擎 Value（教学踏脚石）
│   ├── nn/             #   Module/Linear/激活/损失/Dropout/BatchNorm/
│   │                   #     Conv2d/RNN/LSTM/注意力/Transformer
│   ├── optim/          #   SGD / RMSProp / Adam
│   ├── data/           #   Dataset / DataLoader
│   └── utils/          #   set_seed、gradcheck（数值梯度检查）、load_mnist
├── notebooks/          # 教学 notebook，按 part 分目录，编号即学习顺序
├── tests/              # 对 minitorch 的 pytest（每个算子/层都有梯度检查）
├── scripts/            # 数据下载等辅助脚本
└── data/               # 数据缓存（不入库）
```

> ✅ **项目状态：已完成。** 9 个 Part、26 个 notebook 全部就绪，`minitorch` 框架的每个核心算子/层都通过数值梯度检查（`pytest tests/`，55 项），并与 PyTorch 对照验证。

## 设计理念

- **包是单一事实来源**：能力首次实现总在 notebook 里分步手写并验证；定稿后抽取进 `minitorch/`，后续直接复用。
- **追求理解而非性能**：所有"真训练"都把数据规模 / 轮数控制在 CPU 几分钟内即可跑完；极致性能交给 PyTorch 对照那一半。
- **可复现**：统一使用 `minitorch.set_seed(42)`。

## 致谢与参考

"造轮子"的思路受 Andrej Karpathy 的 micrograd、斯坦福 CS231n、《动手学深度学习》(d2l.ai)、《Deep Learning from Scratch》等优秀资料启发。

## License

MIT
