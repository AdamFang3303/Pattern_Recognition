import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import torch


class NPYShapeDataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        # 标签映射 (根据图片中的文件名调整)
        self.label_map = {
            'cat': 0,
            'circle': 1,
            'dog': 2,
            'hexagon': 3,
            'line': 4,
            'octagon': 5,
            'square': 6,
            'star': 7,
            'triangle': 8
        }

        for f in os.listdir(npy_dir):
            if not f.endswith(".npy"):
                continue

            # 关键！处理图片中的异常文件名
            clean_name = f.lower().replace(" ", "_")  # 处理空格和大小写
            clean_name = clean_name.replace("_npy", "")  # 修正示例中的"date_npy"
            label_name = clean_name.split("_")[-1].split(".")[0]

            if label_name not in self.label_map:
                print(f"⚠️ 跳过无效文件: {f} (无法识别类别 '{label_name}')")
                continue

            file_path = os.path.join(npy_dir, f)
            data = np.load(file_path).reshape(-1, 28, 28)
            self.data.append(data)
            self.labels.extend([self.label_map[label_name]] * len(data))

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.array(self.labels)

        # 打乱数据
        idx = np.random.permutation(len(self.data))
        self.data = self.data[idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.uint8)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, torch.tensor(label, dtype=torch.long)