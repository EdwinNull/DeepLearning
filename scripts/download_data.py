"""一次性下载 MNIST 到 data/（带缓存与校验）。

用法：
    python scripts/download_data.py

之后各 notebook 通过 minitorch.utils.load_mnist 从本地缓存读取，不再重复下载。
"""
import sys


def main():
    try:
        from minitorch.utils import load_mnist
    except Exception as e:  # noqa: BLE001
        print("无法导入 minitorch，请先在项目根目录执行： pip install -e .")
        print("错误：", e)
        return 1

    print("正在下载/校验 MNIST 到 ./data ……")
    (x_tr, y_tr), (x_te, y_te) = load_mnist(root="data")
    print(f"完成 ✅  训练集 {x_tr.shape}，测试集 {x_te.shape}")
    print(f"像素范围 [{x_tr.min():.1f}, {x_tr.max():.1f}]，标签类别 {sorted(set(y_tr.tolist()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
