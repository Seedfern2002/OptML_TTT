from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import random
from src.data_generator import enumerate_states


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
        """Return the number of files in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """Return dataset item at the specified index."""
        file_path = os.path.join(self.save_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True)
        x = torch.tensor(data[0], dtype=torch.float32)
        y = torch.tensor(data[1], dtype=torch.float32)
        return x, y
    
    def get_train_test_split(self, test_size=0.2):
        """Split dataset into train and test files based on the provided ratio."""
        file_names = self.files

        hash_dict = {}
        for name in file_names:
            prefix = int(name.split("_")[0])
            hash_value = int(name.split("_")[1].split(".")[0])
            if prefix not in hash_dict:
                hash_dict[prefix] = []
            hash_dict[prefix].append(hash_value)
        
        for prefix in hash_dict.keys():
            hash_dict[prefix].sort()

        test_hashes = get_test_set_hashes(hash_dict, ratio=test_size)
        test_files = [file for file in file_names if int(file.split("_")[1].split(".")[0]) in test_hashes]
        train_files = [file for file in file_names if file not in test_files]
        print(f'Train/Test split: {len(train_files)} train files, {len(test_files)} test files')
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

def get_test_set_hashes(hash_dict, symmetries_save_dir="symmetries", ratio=0.2):
    """Generate test set hashes from the given hash dictionary."""
    test_hash = set()
    for prefix in hash_dict.keys():
        temp_hash = set()
        hashes = hash_dict[prefix]
        total_hashes = len(hashes)
        num_test_hashes = int(total_hashes * ratio)
        for hash in hashes: 
            symmetries_file = os.path.join(symmetries_save_dir, f"{prefix}_{hash}.npy")
            symmetries = np.load(symmetries_file, allow_pickle=True)
            if len(symmetries) > 0:
                temp_hash.update(symmetries)
            if len(temp_hash) >= num_test_hashes:
                break
        test_hash.update(temp_hash)
    
    return list(test_hash)


def load_dataset(order="easy_to_hard", save_dir="monte_carlo_data", split=None, data_percentage=1.0):
    """Load dataset and return a DataLoader instance."""
    files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")],
                   key=lambda name: int(name.split("_")[0]),
                   reverse=(order == "hard_to_easy"))
    if order == 'random':
        random.shuffle(files)
    
    # Pass data_percentage to the dataset
    return DataLoader(TicTacToeDataset(files, save_dir, split=split, data_percentage=data_percentage), batch_size=32, shuffle=False)
