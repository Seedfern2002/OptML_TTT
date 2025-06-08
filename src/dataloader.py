from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import random


class TicTacToeDataset(Dataset):
    def __init__(self, files, save_dir="monte_carlo_data", split=None):
        self.save_dir = save_dir
        self.files = files
        self.split = split
        if split is not None:
            train_files, test_files = self.get_train_test_split()
            print(f"Split {split}: {len(train_files)} train files, {len(test_files)} test files")
            self.files = train_files if split == 'train' else test_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.save_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True)
        x = torch.tensor(data[0], dtype=torch.float32)
        y = torch.tensor(data[1], dtype=torch.float32)
        return x, y
    
    def get_train_test_split(self, test_size=0.2):
        file_names = self.files
        hash_values = [int(name.split("_")[1].split(".")[0]) for name in file_names]
        # take those idx with hash values less than 10**8 * test_size
        threshold = int(10**8 * test_size)
        test_idx = [i for i, h in enumerate(hash_values) if h < threshold]
        train_idx = [i for i in range(len(file_names)) if i not in test_idx]
        train_files = [file_names[i] for i in train_idx]
        test_files = [file_names[i] for i in test_idx]
        return train_files, test_files



def load_dataset(order="easy_to_hard", save_dir="monte_carlo_data", split=None):
    files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")],
                   key=lambda name: int(name.split("_")[0]),
                   reverse=(order == "hard_to_easy"))
    if order == 'random':
        random.shuffle(files)
    return DataLoader(TicTacToeDataset(files, save_dir, split=split), batch_size=32, shuffle=False)
