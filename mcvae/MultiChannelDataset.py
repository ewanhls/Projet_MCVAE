import torch
from torch.utils.data import Dataset, DataLoader

class MultiChannelDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = [torch.tensor(d, dtype=torch.float32) for d in data_list]
        self.n_samples = self.data_list[0].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return [d[idx] for d in self.data_list]
