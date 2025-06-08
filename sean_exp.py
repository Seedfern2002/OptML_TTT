# import numpy as np
# import torch
# import random
# import os
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F # Needed for softmax if you choose to use it
# from tqdm import tqdm
#
# # Import your modules
# from model.model import TicTacToeCNN
# from src.dataloader import load_dataset
# from src.eval import evaluate_agents
# from src.train import train_model
#
#
# def set_seed(seed):
#     """Sets the random seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed) # For multi-GPU setups
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# if __name__ == "__main__":
#     # Hardcoded training parameters
#     epochs = 10 # Renamed to plural for clarity
#     optimizer_choice = "adam"
#     # criterion_choice = "kl_div" # Or "mse" (assuming you will train with MSE later)
#     criterion_choice = "mse" # Or "mse" (assuming you will train with MSE later)
#                                 # If you plan to switch to MSE, make sure your model output is suitable.
#     
#     # Define seeds for multiple runs
#     seeds = [42, 101, 202, 303, 404] 
#     
#     # Define default games for most evaluations and a reduced amount for MCTS
#     default_eval_games = 5000
#     mcts_eval_games = 100 # Reduced games for MCTS comparison
#
#     all_evaluation_results = {}
#
#     for seed in seeds:
#         print(f"\n--- Running experiment with seed: {seed} ---")
#         set_seed(seed) # Set seed for the current run
#
#         # Define save directory for this seed
#         seed_save_dir = os.path.join("results", f'seed_{seed}_{optimizer_choice}_{criterion_choice}_epoch_{epochs}')
#         os.makedirs(seed_save_dir, exist_ok=True)
#
#         models = {}
#         # Train and save models
#         for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
#             print(f"Loading {curriculum_type} dataset...")
#             data_loader = load_dataset(curriculum_type)
#             print(f"Training {curriculum_type} model...")
#             # Pass the criterion_choice to the model during initialization if its output layer depends on it
#             # (e.g., if kl_div implies log-softmax output vs. mse implying linear output)
#             model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div")) 
#             train_model(model, data_loader, epochs=epochs, optimizer=optimizer_choice, criterion=criterion_choice)
#             
#             model_name = f"model_{curriculum_type.replace('_to_', '_').replace('random', 'random_curriculum')}"
#             torch.save(model.state_dict(), os.path.join(seed_save_dir, f"{model_name}.pth"))
#             model.eval() # Set to eval mode for evaluation
#             models[model_name] = model
#
#         # Store results for this seed
#         current_seed_results = {}
#
#         print(f"\n--- Evaluating Models for Seed {seed} ---")
#         
#         # Comparisons between curriculum models
#         print("\n--- Evaluating Curriculum Models Against Each Other ---")
#         curriculum_model_names = list(models.keys())
#         for i in range(len(curriculum_model_names)):
#             for j in range(i + 1, len(curriculum_model_names)):
#                 name1 = curriculum_model_names[i]
#                 name2 = curriculum_model_names[j]
#                 # Use default_eval_games for model vs model comparisons
#                 # Pass the criterion_choice to evaluate_agents for both agents
#                 results = evaluate_agents(models[name1], models[name2], 
#                                           games=default_eval_games, 
#                                           agent1_criterion=criterion_choice, 
#                                           agent2_criterion=criterion_choice)
#                 comparison_name = f"{name1.replace('model_', '')} vs {name2.replace('model_', '')}"
#                 current_seed_results[comparison_name] = results
#                 print(f"Results ({comparison_name}): {results}")
#
#         # Comparisons with MCTS and Random agents
#         print("\n--- Evaluating Curriculum Models Against MCTS Agent ---")
#         for model_name, model_obj in models.items():
#             # Use mcts_eval_games for MCTS comparisons
#             # Pass the criterion_choice for the CNN model
#             results = evaluate_agents(model_obj, 'mcts_agent', 
#                                       games=mcts_eval_games, 
#                                       agent1_criterion=criterion_choice, 
#                                       agent2_criterion=None) # MCTS doesn't have a criterion
#             comparison_name = f"{model_name.replace('model_', '')} vs MCTS_agent"
#             current_seed_results[comparison_name] = results
#             print(f"Results ({comparison_name}): {results}")
#         
#         print("\n--- Evaluating Curriculum Models Against Pure Random Actions ---")
#         for model_name, model_obj in models.items():
#             # Use default_eval_games for Random agent comparisons
#             # Pass the criterion_choice for the CNN model
#             results = evaluate_agents(model_obj, 'random_agent', 
#                                       games=default_eval_games, 
#                                       agent1_criterion=criterion_choice, 
#                                       agent2_criterion=None) # Random doesn't have a criterion
#             comparison_name = f"{model_name.replace('model_', '')} vs Random_agent"
#             current_seed_results[comparison_name] = results
#             print(f"Results ({comparison_name}): {results}")
#
#
#         all_evaluation_results[seed] = current_seed_results
#
#         # Log results for the current seed to a file
#         with open(os.path.join(seed_save_dir, "evaluation_results.txt"), "w") as f:
#             f.write(f"--- Evaluation Results for Seed: {seed} ---\n\n")
#             for comparison, res in current_seed_results.items():
#                 f.write(f"Results ({comparison}): {res}\n")
#             f.write("\n")
#
#     print("\n--- All Experiments Complete ---")
#     # You can add logic here to average results across seeds or perform further analysis
#     # For example, print overall average wins/draws if needed.
#     
#     print("\nSummary of all results (per seed):")
#     for seed, results_dict in all_evaluation_results.items():
#         print(f"\nSeed {seed}:")
#         for comparison, results in results_dict.items():
#             print(f"  {comparison}: {results}")
import numpy as np
import torch
import random
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Needed for softmax if you choose to use it
from tqdm import tqdm

# Import your modules
from model.model import TicTacToeCNN
from src.dataloader import load_dataset
from src.eval import evaluate_agents
from src.train import train_model


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_mcts_data(data_dir="../monte_carlo_data"):
    """
    Loads pre-generated MCTS data from .npy files.
    Returns a dictionary mapping board state string to MCTS probabilities.
    """
    mcts_data = {}
    print(f"Loading MCTS data from {data_dir}...")
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".npy"):
            file_path = os.path.join(data_dir, filename)
            data = np.load(file_path, allow_pickle=True)
            board_input = data[0]
            mcts_probs = data[1]

            # Reconstruct the board state string from the input array
            board_list = [' ' for _ in range(9)]
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if board_input[0, i, j] == 1:
                        board_list[idx] = 'O'
                    elif board_input[1, i, j] == 1:
                        board_list[idx] = 'X'
            state_key = ''.join(board_list)
            mcts_data[state_key] = mcts_probs
    print(f"Loaded {len(mcts_data)} MCTS states.")
    return mcts_data


if __name__ == "__main__":
    # Hardcoded training parameters
    epochs = 10 # Renamed to plural for clarity
    optimizer_choice = "adam"
    criterion_choice = "mse" # Or "kl_div"

    # Define seeds for multiple runs
    seeds = [42, 101, 202, 303, 404]

    # Define default games for most evaluations and a reduced amount for MCTS
    default_eval_games = 5000
    mcts_eval_games = 100 # Reduced games for MCTS comparison

    all_evaluation_results = {}

    # Load MCTS data once at the beginning
    preloaded_mcts_data = load_mcts_data()

    for seed in seeds:
        print(f"\n--- Running experiment with seed: {seed} ---")
        set_seed(seed) # Set seed for the current run

        # Define save directory for this seed
        seed_save_dir = os.path.join("results", f'seed_{seed}_{optimizer_choice}_{criterion_choice}_epoch_{epochs}')
        os.makedirs(seed_save_dir, exist_ok=True)

        models = {}
        # Train and save models
        for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
            print(f"Loading {curriculum_type} dataset...")
            data_loader = load_dataset(curriculum_type)
            print(f"Training {curriculum_type} model...")
            model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
            train_model(model, data_loader, epochs=epochs, optimizer=optimizer_choice, criterion=criterion_choice)

            model_name = f"model_{curriculum_type.replace('_to_', '_').replace('random', 'random_curriculum')}"
            torch.save(model.state_dict(), os.path.join(seed_save_dir, f"{model_name}.pth"))
            model.eval() # Set to eval mode for evaluation
            models[model_name] = model

        # Store results for this seed
        current_seed_results = {}

        print(f"\n--- Evaluating Models for Seed {seed} ---")

        # Comparisons between curriculum models
        print("\n--- Evaluating Curriculum Models Against Each Other ---")
        curriculum_model_names = list(models.keys())
        for i in range(len(curriculum_model_names)):
            for j in range(i + 1, len(curriculum_model_names)):
                name1 = curriculum_model_names[i]
                name2 = curriculum_model_names[j]
                results = evaluate_agents(models[name1], models[name2],
                                          games=default_eval_games,
                                          agent1_criterion=criterion_choice,
                                          agent2_criterion=criterion_choice)
                comparison_name = f"{name1.replace('model_', '')} vs {name2.replace('model_', '')}"
                current_seed_results[comparison_name] = results
                print(f"Results ({comparison_name}): {results}")

        # Comparisons with MCTS (using pre-computed data) and Random agents
        print("\n--- Evaluating Curriculum Models Against Pre-computed MCTS Agent ---")
        for model_name, model_obj in models.items():
            results = evaluate_agents(model_obj, 'mcts_data_agent', # Changed to 'mcts_data_agent'
                                      games=mcts_eval_games,
                                      agent1_criterion=criterion_choice,
                                      agent2_criterion=None, # MCTS doesn't have a criterion
                                      mcts_data=preloaded_mcts_data) # Pass the loaded data
            comparison_name = f"{model_name.replace('model_', '')} vs MCTS_data_agent"
            current_seed_results[comparison_name] = results
            print(f"Results ({comparison_name}): {results}")

        print("\n--- Evaluating Curriculum Models Against Pure Random Actions ---")
        for model_name, model_obj in models.items():
            results = evaluate_agents(model_obj, 'random_agent',
                                      games=default_eval_games,
                                      agent1_criterion=criterion_choice,
                                      agent2_criterion=None) # Random doesn't have a criterion
            comparison_name = f"{model_name.replace('model_', '')} vs Random_agent"
            current_seed_results[comparison_name] = results
            print(f"Results ({comparison_name}): {results}")


        all_evaluation_results[seed] = current_seed_results

        # Log results for the current seed to a file
        with open(os.path.join(seed_save_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"--- Evaluation Results for Seed: {seed} ---\n\n")
            for comparison, res in current_seed_results.items():
                f.write(f"Results ({comparison}): {res}\n")
            f.write("\n")

    print("\n--- All Experiments Complete ---")
    print("\nSummary of all results (per seed):")
    for seed, results_dict in all_evaluation_results.items():
        print(f"\nSeed {seed}:")
        for comparison, results in results_dict.items():
            print(f"  {comparison}: {results}")
