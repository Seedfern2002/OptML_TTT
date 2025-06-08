import numpy as np
import torch
import random
from src.mcts import monte_carlo_probs # Import MCTS
from src.tictactoe import TicTacToe # Import TicTacToe


def select_move(agent, game):
    """
    Selects a move based on the agent's type.

    Args:
        agent: Can be a trained CNN model, 'mcts_agent', or 'random_agent'.
        game: The current TicTacToe game state.

    Returns:
        The chosen move (integer from 0-8).
    """
    if isinstance(agent, torch.nn.Module): # CNN model
        board = np.zeros((2, 3, 3))
        for i, v in enumerate(game.board):
            if v == 'X':
                board[0, i // 3, i % 3] = 1
            elif v == 'O':
                board[1, i // 3, i % 3] = 1
        with torch.no_grad():
            # Ensure the model output is correctly reshaped for 3x3 board
            # probs = agent(torch.tensor([board], dtype=torch.float32)).squeeze().view(9).numpy() # for mse
            probs = torch.exp(agent(torch.tensor([board], dtype=torch.float32)).squeeze()).view(9).numpy() # for kl divergence

        legal = game.available_moves()
        # Filter probabilities for legal moves and re-normalize
        filtered_probs = np.zeros(9)
        for i in legal:
            filtered_probs[i] = probs[i]

        s = filtered_probs.sum()
        if s == 0: # If all legal moves have 0 probability (e.g., due to ReLU or initial state)
            return random.choice(legal)

        normalized_probs = filtered_probs / s
        return np.random.choice(range(9), p=normalized_probs)

    elif agent == 'mcts_agent':
        # MCTS returns a 3x3 numpy array of probabilities, flatten and select
        probs_3x3 = monte_carlo_probs(game, simulations=10000)
        probs = probs_3x3.flatten()
        legal = game.available_moves()

        filtered_probs = np.zeros(9)
        for i in legal:
            filtered_probs[i] = probs[i]

        s = filtered_probs.sum()
        if s == 0:
            return random.choice(legal)

        normalized_probs = filtered_probs / s
        return np.random.choice(range(9), p=normalized_probs)

    elif agent == 'random_agent':
        return random.choice(game.available_moves())
    else:
        raise ValueError("Unknown agent type provided to select_move.")


def evaluate_agents(agent1, agent2, games=5000):
    """
    Evaluates two agents playing Tic-Tac-Toe against each other.

    Args:
        agent1: The first agent (a trained CNN model, 'mcts_agent', or 'random_agent').
        agent2: The second agent (a trained CNN model, 'mcts_agent', or 'random_agent').
        games: Number of games to play.

    Returns:
        A dictionary with win counts for agent1, agent2, and draws.
    """
    results = {"agent1_wins": 0, "agent2_wins": 0, "draw": 0}
    
    for i in range(games):
        game = TicTacToe()
        # Alternate who goes first
        players = [agent1, agent2] if i % 2 == 0 else [agent2, agent1]
        
        while game.winner() is None:
            # Determine which player's turn it is and select their corresponding agent
            current_player_agent = players[0] if game.current_player == 'X' else players[1]
            move = select_move(current_player_agent, game)
            game.make_move(move)
            
        result = game.winner()
        
        if result == 'X':
            if i % 2 == 0: # Agent1 was 'X'
                results["agent1_wins"] += 1
            else: # Agent2 was 'X'
                results["agent2_wins"] += 1
        elif result == 'O':
            if i % 2 == 0: # Agent2 was 'O'
                results["agent2_wins"] += 1
            else: # Agent1 was 'O'
                results["agent1_wins"] += 1
        else: # Draw
            results["draw"] += 1
    return results
