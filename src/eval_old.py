import numpy as np
import torch
import random

def evaluate_models(model1, model2, games=5000):
    from .tictactoe import TicTacToe

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
