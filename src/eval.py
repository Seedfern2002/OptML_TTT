import numpy as np
import torch
import random
from model import TicTacToeCNN
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

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

def train_model(model, dataloader, epochs=1, optimizer="adam", criterion="mse"):
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")
    
    if criterion == "mse":
        loss_fn = nn.MSELoss()
    elif criterion == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported criterion. Use 'mse' or 'cross_entropy'.")
    # use kl divergence loss for probabilities
    # loss_fn = nn.KLDivLoss()
    # TODO: try different loss func later
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def evaluate_models(model1, model2, games=5000):
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
        probs = np.array([probs[i//3][i % 3] if i in legal else 0 for i in range(9)])
        s = probs.sum()
        if s == 0:
            return random.choice(legal)
        probs = probs / s
        # print(f'sum of probs: {sum(probs)}')
        # use greedy policy
        # return the largest probability move
        # return np.argmax(probs)
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
    return results


if __name__ == "__main__":
    # set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    parser = argparse.ArgumentParser(description="Train and evaluate TicTacToe models.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory where dataset is stored.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam", help="Optimizer to use for training.")
    parser.add_argument("--criterion", type=str, choices=["mse", "cross_entropy"], default="mse", help="Loss function to use for training.")
    args = parser.parse_args()
    
    epoch = args.epochs
    save_dir = args.save_dir
    optimizer_choice = args.optimizer
    criterion_choice = args.criterion
    save_dir = os.path.join(save_dir, f'{optimizer_choice}_{criterion_choice}_epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    print("Loading easy-to-hard dataset...")
    easy_loader = load_dataset("easy_to_hard")
    print("Training easy-to-hard model...")
    model_easy = TicTacToeCNN()
    train_model(model_easy, easy_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice)
    torch.save(model_easy.state_dict(), os.path.join(save_dir, "model_easy.pth"))

    print("Loading hard-to-easy dataset...")
    hard_loader = load_dataset("hard_to_easy")
    print("Training hard-to-easy model...")
    model_hard = TicTacToeCNN()
    train_model(model_hard, hard_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice)
    torch.save(model_hard.state_dict(), os.path.join(save_dir, "model_hard.pth"))

    print("Loading random dataset...")
    random_loader = load_dataset("random")
    print("Training random model...")
    model_random = TicTacToeCNN()
    train_model(model_random, random_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice)
    torch.save(model_random.state_dict(), os.path.join(save_dir, "model_random.pth"))

    print("Evaluating models...")
    print("Evaluating easy model against hard model...")
    results_evh = evaluate_models(model_easy, model_hard)
    print("Results (easy vs hard):", results_evh)
    print("Evaluating easy model against random model...")
    results_evr = evaluate_models(model_easy, model_random)
    print("Results (easy vs random):", results_evr)
    print("Evaluating hard model against random model...")
    results_hvr = evaluate_models(model_hard, model_random)
    print("Results (hard vs random):", results_hvr)

    # log the results to a file
    with open(os.path.join(save_dir, "evaluation_results.txt"), "w") as f:
        f.write("Results (easy vs hard): " + str(results_evh) + "\n")
        f.write("Results (easy vs random): " + str(results_evr) + "\n")
        f.write("Results (hard vs random): " + str(results_hvr) + "\n")


    # model_easy = TicTacToeCNN()
    # model_easy.load_state_dict(torch.load("model_easy.pth"))
    # model_easy.eval()

    # model_hard = TicTacToeCNN()
    # model_hard.load_state_dict(torch.load("model_hard.pth"))
    # model_hard.eval()

    # print with input of all 0
    # prob_easy = model_easy(torch.zeros((1, 2, 3, 3), dtype=torch.float32))
    # prob_hard = model_hard(torch.zeros((1, 2, 3, 3), dtype=torch.float32))
    # print("Normalized probabilities for model_easy:", prob_easy.detach().numpy().flatten() / sum(prob_easy.detach().numpy().flatten()))
    # print("Normalized probabilities for model_hard:", prob_hard.detach().numpy().flatten() / sum(prob_hard.detach().numpy().flatten()))
