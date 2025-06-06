import numpy as np
import os
import random
from tictactoe import TicTacToe

def random_playout(game):
    while not game.winner():
        move = random.choice(game.available_moves())
        game.make_move(move)
    return game.winner()

def monte_carlo_probs(game, simulations=10000):
    probs = np.zeros(9)
    for move in game.available_moves():
        wins = 0
        for _ in range(simulations):
            sim_game = game.clone()
            sim_game.make_move(move)
            result = random_playout(sim_game)
            if result == game.current_player:
                wins += 1
            elif result == "Draw":
                wins += 0.5
        probs[move] = wins / simulations
    return probs.reshape(3, 3)

def board_to_input(board, player):
    arr = np.zeros((2, 3, 3))
    for i, v in enumerate(board):
        if v == 'X':
            arr[0, i // 3, i % 3] = 1
        elif v == 'O':
            arr[1, i // 3, i % 3] = 1
    return arr if player == 'X' else arr[::-1]

def generate_data(game=None, visited=set()):
    if game is None:
        game = TicTacToe()

    state_key = "".join(game.board)
    if state_key in visited:
        return
    visited.add(state_key)

    if game.winner() is not None:
        return

    x = board_to_input(game.board, game.current_player)
    y = monte_carlo_probs(game)
    filename = f"{sum(1 for c in game.board if c != ' ')}_{hash(state_key)%10**8}.npy"
    save_path = os.path.join("monte_carlo_data", filename)
    
    np.save(save_path, np.array([x, y], dtype=object))

    for move in game.available_moves():
        new_game = game.clone()
        new_game.make_move(move)
        generate_data(new_game, visited)

if __name__ == "__main__":
    os.makedirs("monte_carlo_data", exist_ok=True)
    generate_data()