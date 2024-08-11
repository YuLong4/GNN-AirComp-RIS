import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 保存路径和预处理操作
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        # 从 raw 目录中加载数据
        data = np.load(self.data_dir)  # 加载数据
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
