import numpy as np
import os
from src.tictactoe import TicTacToe
from src.mcts import monte_carlo_probs
from multiprocessing import Pool, cpu_count
import random


def board_to_input(board, player):
    arr = np.zeros((2, 3, 3))
    for i, v in enumerate(board):
        if v == 'X':
            arr[0, i // 3, i % 3] = 1
        elif v == 'O':
            arr[1, i // 3, i % 3] = 1
    return arr if player == 'X' else arr[::-1]

def get_all_symmetries(board):
    """
    Given a board (list of 9 characters in row-major order),
    return a list of all symmetric representations as strings.
    The function considers 90° rotations, horizontal reflections, and vertical reflections.
    """
    mat = [board[i*3:(i+1)*3] for i in range(3)]
    
    def rotate(m):
        return [[m[2-j][i] for j in range(3)] for i in range(3)]
    
    def horizontal_reflect(m):
        return [row[::-1] for row in m]
    
    def vertical_reflect(m):
        return m[::-1]
    
    transforms = []
    current = mat
    for _ in range(4):
        transforms.append(''.join(''.join(row) for row in current))
        transforms.append(''.join(''.join(row) for row in horizontal_reflect(current)))
        transforms.append(''.join(''.join(row) for row in vertical_reflect(current)))
        current = rotate(current)
        
    return list(set(transforms))

def process_state(state):
    """
    Compute and save Monte Carlo data for a given state.
    state: tuple(board_list, current_player)
    """
    board, player = state
    game = TicTacToe()
    game.board = board
    game.current_player = player

    # skip terminal states
    if game.winner() is not None:
        return

    x = board_to_input(game.board, game.current_player)
    y = monte_carlo_probs(game)
    state_key = ''.join(game.board)
    filename = f"{sum(1 for c in game.board if c != ' ')}_{abs(hash(state_key)) % 10**8}.npy"
    save_dir = "./monte_carlo_data"
    symmetries_save_dir = os.path.join("symmetries")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(symmetries_save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, np.array([x, y], dtype=object))
    # save hashes of the symetries
    symmetries = get_all_symmetries(game.board)
    hashes = [abs(hash(sym)) % 10**8 for sym in symmetries]
    symmetries_save_path = os.path.join(symmetries_save_dir, filename)
    np.save(symmetries_save_path, np.array(hashes, dtype=object))
    

def enumerate_states(game=None, visited=None):
    """
    Enumerate all possible states of Tic-Tac-Toe.
    """
    if visited is None:
        visited = set()
    if game is None:
        game = TicTacToe()
    key = ''.join(game.board) + game.current_player
    if key in visited:
        return []
    visited.add(key)
    states = [(game.board[:], game.current_player)]
    if game.winner() is not None:
        return states
    for move in game.available_moves():
        new_game = game.clone()
        new_game.make_move(move)
        states += enumerate_states(new_game, visited)
    return states


if __name__ == "__main__":
    np.random.seed(42)  
    random.seed(42)
    os.makedirs("./monte_carlo_data", exist_ok=True)
    # collect all unique states (which includes terminal states)
    print("Enumerating states...")
    states = enumerate_states()
    # parallel processing only on non-terminal states
    print(f"Starting data generation using {cpu_count()} cores...")
    with Pool(cpu_count()) as pool:
        pool.map(process_state, states)
    print("Data generation complete.")
