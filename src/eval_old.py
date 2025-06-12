import numpy as np
import torch
import random
import os
from src.tictactoe import TicTacToe
from model.model import TicTacToeCNN
from tqdm import tqdm

def evaluate_models(model1, model2, games=5000):
    def select_move(model, game, kl_div=False):
        board = np.zeros((2, 3, 3))
        for i, v in enumerate(game.board):
            if v == 'X':
                board[0, i // 3, i % 3] = 1
            elif v == 'O':
                board[1, i // 3, i % 3] = 1
        with torch.no_grad():
            input = torch.tensor([board], dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor([board], dtype=torch.float32)
            probs = model(input).cpu().squeeze().numpy()
        if kl_div:
            probs = np.exp(probs)
        legal = game.available_moves()
        probs = np.array([probs[i//3][i % 3] if i in legal else 0 for i in range(9)])
        
        s = probs.sum()
        if s == 0:
            return random.choice(legal)
        probs = probs / s
        
        return np.random.choice(range(9), p=probs)

    results = {"model1": 0, "model2": 0, "draw": 0}
    model1.eval()
    model2.eval()
    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()
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

def eval_models_all_epochs(save_dir, order1, order2, games=5000, per_epochs = 1):
    order1 = 'model_' + order1
    order2 = 'model_' + order2
    models_order1 = os.listdir(os.path.join(save_dir, order1))
    models_order2 = os.listdir(os.path.join(save_dir, order2))

    epochs = sorted(set(int(model.split('_')[-1].split('.')[0]) for model in models_order1 if model.endswith('.pth')))
    # only keep epochs that are multiples of per_epochs
    epochs = [epoch for epoch in epochs if (epoch - 1) % per_epochs == 0]
    results = {epoch: {order1: 0, order2: 0, "draw": 0} for epoch in epochs}
    for epoch in tqdm(epochs):
        model1_path = os.path.join(save_dir, order1, f"model_epoch_{epoch}.pth")
        model2_path = os.path.join(save_dir, order2, f"model_epoch_{epoch}.pth")
        
        if not os.path.exists(model1_path) or not os.path.exists(model2_path):
            print(f"Skipping epoch {epoch} as one of the models does not exist: {model1_path} or {model2_path}")
            continue
        
        model1 = TicTacToeCNN(kl_div=('kl_div' in save_dir))
        model2 = TicTacToeCNN(kl_div=('kl_div' in save_dir))
        model1.load_state_dict(torch.load(model1_path))
        model2.load_state_dict(torch.load(model2_path))

        results_epoch = evaluate_models(model1, model2, games=games)
        results[epoch][order1] = results_epoch["model1"]
        results[epoch][order2] = results_epoch["model2"]
        results[epoch]["draw"] = results_epoch["draw"]
    return results
