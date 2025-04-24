# check_data.py
import os
import numpy as np
from config import Config
from data import NPYShapeDataset

def verify_dataset():
    # 检查文件命名
    valid_keys = [ 'circle',
                    'square',
                    'triangle',
                    'cat',
                    'dog',
                    'hexagon',
                    'octagon',
                    'line',
                    'star',]

    for f in os.listdir(Config.npy_dir):
        if not f.endswith('.npy'):
            continue

        # 提取标签名
        label_name = f.split('_')[-1].split('.')[0]
        if label_name not in valid_keys:
            print(f"❌ 非法文件名: {f} (未定义类别 '{label_name}')")
            return False

    # 检查数据平衡性
    counts = {}
    dataset = NPYShapeDataset(Config.npy_dir)
    unique, counts = np.unique(dataset.labels, return_counts=True)
    print("类别分布:")
    for cls, cnt in zip(unique, counts):
        print(f"Class {cls}: {cnt} samples")

    return True


if __name__ == "__main__":
    if verify_dataset():
        print("✅ 数据集验证通过")