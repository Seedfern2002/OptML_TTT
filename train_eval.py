import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from model import TicTacToeCNN

class TicTacToeDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        x = torch.tensor(data[0], dtype=torch.float32)
        y = torch.tensor(data[1], dtype=torch.float32)
        return x, y

def load_dataset(order="easy_to_hard"):
    files = sorted([f for f in os.listdir(".") if f.endswith(".npy")],
                   key=lambda name: int(name.split("_")[0]),
                   reverse=(order == "hard_to_easy"))
    return DataLoader(TicTacToeDataset(files), batch_size=32, shuffle=True)

def train_model(model, dataloader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def evaluate_models(model1, model2, games=100):
    from tictactoe import TicTacToe
    def select_move(model, game):
        board = np.zeros((2, 3, 3))
        for i, v in enumerate(game.board):
            if v == 'X':
                board[0, i // 3, i % 3] = 1
            elif v == 'O':
                board[1, i // 3, i % 3] = 1
        with torch.no_grad():
            probs = model(torch.tensor([board], dtype=torch.float32)).squeeze().numpy()
        legal = game.available_moves()
        probs = [probs[i//3][i%3] if i in legal else 0 for i in range(9)]
        s = sum(probs)
        if s == 0:
            return random.choice(legal)
        probs = [p/s for p in probs]
        return np.random.choice(range(9), p=probs)

    results = {"model1": 0, "model2": 0, "draw": 0}
    for i in range(games):
        game = TicTacToe()
        players = [model1, model2] if i % 2 == 0 else [model2, model1]
        while game.winner() is None:
            move = select_move(players[0 if game.current_player == 'X' else 1], game)
            game.make_move(move)
        result = game.winner()
        if result == 'X':
            results["model1" if i % 2 == 0 else "model2"] += 1
        elif result == 'O':
            results["model2" if i % 2 == 0 else "model1"] += 1
        else:
            results["draw"] += 1
    print(results)