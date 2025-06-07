from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import random


class TicTacToeDataset(Dataset):
    def __init__(self, files, save_dir="monte_carlo_data"):
        self.save_dir = save_dir
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.save_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True)
        x = torch.tensor(data[0], dtype=torch.float32)
        y = torch.tensor(data[1], dtype=torch.float32)
        return x, y


def load_dataset(order="easy_to_hard", save_dir="monte_carlo_data"):
    files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")],
                   key=lambda name: int(name.split("_")[0]),
                   reverse=(order == "hard_to_easy"))
    if order == 'random':
        random.shuffle(files)
    return DataLoader(TicTacToeDataset(files, save_dir), batch_size=32, shuffle=False)
