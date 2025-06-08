import numpy as np
import torch
import random
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


if __name__ == "__main__":
    # Hardcoded training parameters
    epochs = 10 # Renamed to plural for clarity
    optimizer_choice = "adam"
    criterion_choice = "kl_div" # Or "cross_entropy", "kl_div"
    
    # Define seeds for multiple runs
    seeds = [42, 101, 202, 303, 404] 
    
    # Define default games for most evaluations and a reduced amount for MCTS
    default_eval_games = 5000
    mcts_eval_games = 100 # Reduced games for MCTS comparison

    all_evaluation_results = {}

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
                # Use default_eval_games for model vs model comparisons
                results = evaluate_agents(models[name1], models[name2], games=default_eval_games)
                comparison_name = f"{name1.replace('model_', '')} vs {name2.replace('model_', '')}"
                current_seed_results[comparison_name] = results
                print(f"Results ({comparison_name}): {results}")

        # Comparisons with MCTS and Random agents
        print("\n--- Evaluating Curriculum Models Against MCTS Agent ---")
        for model_name, model_obj in models.items():
            # Use mcts_eval_games for MCTS comparisons
            results = evaluate_agents(model_obj, 'mcts_agent', games=mcts_eval_games)
            comparison_name = f"{model_name.replace('model_', '')} vs MCTS_agent"
            current_seed_results[comparison_name] = results
            print(f"Results ({comparison_name}): {results}")
        
        print("\n--- Evaluating Curriculum Models Against Pure Random Actions ---")
        for model_name, model_obj in models.items():
            # Use default_eval_games for Random agent comparisons
            results = evaluate_agents(model_obj, 'random_agent', games=default_eval_games)
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
    # You can add logic here to average results across seeds or perform further analysis
    # For example, print overall average wins/draws if needed.
    
    print("\nSummary of all results (per seed):")
    for seed, results_dict in all_evaluation_results.items():
        print(f"\nSeed {seed}:")
        for comparison, results in results_dict.items():
            print(f"  {comparison}: {results}")
