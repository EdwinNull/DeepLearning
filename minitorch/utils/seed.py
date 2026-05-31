"""统一随机种子工具。

深度学习实验里，"可复现"非常重要：同样的代码、同样的种子，应当得到同样的结果，
这样我们才能可靠地比较"改了一处之后效果有没有变化"。本模块把 random / numpy
（以及 torch，若已安装）的随机种子一次性设好。
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> int:
    """设定 ``random`` / ``numpy``（以及 ``torch``，若已安装）的随机种子。

    Args:
        seed: 随机种子，默认 42。

    Returns:
        实际使用的种子（方便链式写法 ``s = set_seed()``）。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch 是可选依赖，仅在「PyTorch 对照」章节用到
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed
