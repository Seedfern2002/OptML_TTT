import numpy as np
import random


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
