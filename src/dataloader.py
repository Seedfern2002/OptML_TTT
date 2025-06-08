# from torch.utils.data import Dataset, DataLoader
# import os
# import numpy as np
# import torch
# import random
#
#
# class TicTacToeDataset(Dataset):
#     def __init__(self, files, save_dir="monte_carlo_data", split=None):
#         self.save_dir = save_dir
#         self.files = files
#         self.split = split
#         if split is not None:
#             train_files, test_files = self.get_train_test_split()
#             print(f"Split {split}: {len(train_files)} train files, {len(test_files)} test files")
#             self.files = train_files if split == 'train' else test_files
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         file_path = os.path.join(self.save_dir, self.files[idx])
#         data = np.load(file_path, allow_pickle=True)
#         x = torch.tensor(data[0], dtype=torch.float32)
#         y = torch.tensor(data[1], dtype=torch.float32)
#         return x, y
#
#     def get_train_test_split(self, test_size=0.2):
#         file_names = self.files
#         hash_values = [int(name.split("_")[1].split(".")[0]) for name in file_names]
#         # take those idx with hash values less than 10**8 * test_size
#         threshold = int(10**8 * test_size)
#         test_idx = [i for i, h in enumerate(hash_values) if h < threshold]
#         train_idx = [i for i in range(len(file_names)) if i not in test_idx]
#         train_files = [file_names[i] for i in train_idx]
#         test_files = [file_names[i] for i in test_idx]
#         return train_files, test_files
#
#
# def load_dataset(order="easy_to_hard", save_dir="monte_carlo_data", split=None):
#     files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")],
#                    key=lambda name: int(name.split("_")[0]),
#                    reverse=(order == "hard_to_easy"))
#     if order == 'random':
#         random.shuffle(files)
#     return DataLoader(TicTacToeDataset(files, save_dir, split=split), batch_size=32, shuffle=False)
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import random


class TicTacToeDataset(Dataset):
    def __init__(self, files, save_dir="monte_carlo_data", split=None, data_percentage=1.0):
        self.save_dir = save_dir
        self.files = files
        self.split = split
        if split is not None:
            train_files, test_files = self.get_train_test_split()
            # print(f"Split {split}: {len(train_files)} train files, {len(test_files)} test files")
            self.files = train_files if split == 'train' else test_files

        # Apply data percentage subsampling after train/test split
        if data_percentage < 1.0:
            self.files = self.subsample_by_difficulty(self.files, data_percentage)

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

    def subsample_by_difficulty(self, files, percentage):
        """
        Subsamples files based on difficulty (number of moves made).
        Files are grouped by difficulty, and then a percentage is taken from each group.
        """
        if not (0.0 <= percentage <= 1.0):
            raise ValueError("Percentage must be between 0 and 1.")

        # Group files by the number of moves (difficulty)
        difficulty_groups = {}
        for f in files:
            num_moves = int(f.split("_")[0])
            if num_moves not in difficulty_groups:
                difficulty_groups[num_moves] = []
            difficulty_groups[num_moves].append(f)

        subsampled_files = []
        for num_moves, file_list in difficulty_groups.items():
            # Randomly sample 'percentage' of files from each difficulty group
            num_to_sample = max(1, int(len(file_list) * percentage)) # Ensure at least one file if percentage > 0
            subsampled_files.extend(random.sample(file_list, num_to_sample))
        
        # Sort files to maintain original curriculum order if applicable, otherwise random
        # This sorting logic should align with load_dataset's ordering
        subsampled_files.sort(key=lambda name: int(name.split("_")[0])) 
        return subsampled_files


def load_dataset(order="easy_to_hard", save_dir="monte_carlo_data", split=None, data_percentage=1.0):
    files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")],
                   key=lambda name: int(name.split("_")[0]),
                   reverse=(order == "hard_to_easy"))
    if order == 'random':
        random.shuffle(files)
    
    # Pass data_percentage to the dataset
    return DataLoader(TicTacToeDataset(files, save_dir, split=split, data_percentage=data_percentage), batch_size=32, shuffle=False)
