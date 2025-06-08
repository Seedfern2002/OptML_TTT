import numpy as np
import torch
import random
from src.mcts import monte_carlo_probs # Import MCTS
from src.tictactoe import TicTacToe # Import TicTacToe


def select_move(agent, game, criterion_choice=None, mcts_data=None): # Add mcts_data parameter
    """
    Selects a move based on the agent's type.

    Args:
        agent: Can be a trained CNN model, 'mcts_agent', 'mcts_data_agent', or 'random_agent'.
        game: The current TicTacToe game state.
        criterion_choice: The loss function used for training the CNN model ('mse' or 'kl_div').
                          Required when agent is a CNN model.
        mcts_data: Pre-computed Monte Carlo probabilities. Required when agent is 'mcts_data_agent'.

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
            raw_output = agent(torch.tensor([board], dtype=torch.float32)).squeeze().view(9)

            if criterion_choice == "mse":
                probs = raw_output.numpy()
            elif criterion_choice == "kl_div":
                probs = torch.exp(raw_output).numpy()
            else:
                raise ValueError("criterion_choice must be 'mse' or 'kl_div' when agent is a CNN model.")

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

    elif agent == 'mcts_data_agent': # New agent type to use pre-computed MCTS data
        if mcts_data is None:
            raise ValueError("mcts_data must be provided when agent is 'mcts_data_agent'.")

        state_key = ''.join(game.board)
        # Check if the board state exists in the MCTS data
        if state_key in mcts_data:
            probs_3x3 = mcts_data[state_key].reshape(3, 3) # MCTS data stores 3x3 array
        else:
            # Fallback to random play if data for this state is not found
            # This should ideally not happen if all states are pre-generated
            print(f"Warning: MCTS data not found for state {state_key}. Falling back to random move.")
            return random.choice(game.available_moves())

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


def evaluate_agents(agent1, agent2, games=5000, agent1_criterion=None, agent2_criterion=None, mcts_data=None): # Add mcts_data
    """
    Evaluates two agents playing Tic-Tac-Toe against each other.

    Args:
        agent1: The first agent (a trained CNN model, 'mcts_agent', 'mcts_data_agent', or 'random_agent').
        agent2: The second agent (a trained CNN model, 'mcts_agent', 'mcts_data_agent', or 'random_agent').
        games: Number of games to play.
        agent1_criterion: The loss function used for training agent1 if it's a CNN model.
        agent2_criterion: The loss function used for training agent2 if it's a CNN model.
        mcts_data: Pre-computed Monte Carlo probabilities. Passed to select_move if an MCTS agent is used.

    Returns:
        A dictionary with win counts for agent1, agent2, and draws.
    """
    results = {"agent1_wins": 0, "agent2_wins": 0, "draw": 0}

    for i in range(games):
        game = TicTacToe()
        # Alternate who goes first
        players = [(agent1, agent1_criterion), (agent2, agent2_criterion)] if i % 2 == 0 else [(agent2, agent2_criterion), (agent1, agent1_criterion)]

        while game.winner() is None:
            # Determine which player's turn it is and select their corresponding agent
            current_player_agent_info = players[0] if game.current_player == 'X' else players[1]
            current_player_agent, current_agent_criterion = current_player_agent_info # CORRECTED LINE HERE!

            # Pass the criterion_choice and mcts_data to select_move based on agent type
            if isinstance(current_player_agent, torch.nn.Module):
                move = select_move(current_player_agent, game, criterion_choice=current_agent_criterion)
            elif current_player_agent == 'mcts_data_agent': # Use mcts_data for this agent type
                move = select_move(current_player_agent, game, mcts_data=mcts_data)
            else:
                move = select_move(current_player_agent, game) # For mcts_agent (if you still want to run it) or random_agent

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
