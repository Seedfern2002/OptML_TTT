import numpy as np
import torch
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your modules
from model.model import TicTacToeCNN
from src.dataloader import load_dataset
from src.eval import evaluate_agents
from src.train import train_model, train_model_with_early_stopping


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


def load_mcts_data(data_dir="./monte_carlo_data"):
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
    epochs = 5 # Renamed to plural for clarity
    optimizer_choice = "adam"
    criterion_choice = "mse" # Or "kl_div"

    # Define seeds for multiple runs
    seeds = [42, 101, 202, 303, 404]

    # Define default games for most evaluations and a reduced amount for MCTS
    default_eval_games = 5000
    mcts_eval_games = 5000 # Reduced games for MCTS comparison

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
            # Use train_dataloader and test_dataloader for more detailed logging
            train_data_loader = load_dataset(curriculum_type)
            print(f"Training {curriculum_type} model...")
            model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
            train_model(model, train_data_loader, epochs=epochs, optimizer=optimizer_choice, criterion=criterion_choice, disable_wandb=True, verbose=False)

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

    # --- New Experiment: Training on different data portions ---
    print("\n--- Running Data Portion Experiment ---")
    data_percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
    data_portion_results = {curriculum_type: {p: [] for p in data_percentages} for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]}

    for seed in seeds:
        print(f"\n--- Data Portion Experiment with seed: {seed} ---")
        set_seed(seed)

        portion_save_dir = os.path.join("results", f'data_portion_seed_{seed}_{optimizer_choice}_{criterion_choice}_epoch_{epochs}')
        os.makedirs(portion_save_dir, exist_ok=True)

        for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
            # Load fixed test set once per curriculum type
            test_data_loader = load_dataset(curriculum_type, split='test', data_percentage=1.0)

            for percentage in data_percentages:
                print(f"Training {curriculum_type} model with {percentage*100}% of training data...")
                
                # Load training set with specified percentage
                train_data_loader = load_dataset(curriculum_type, split='train', data_percentage=percentage)

                model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
                
                # Use train_model_with_early_stopping for the data portion experiment
                train_model_with_early_stopping(
                    model, train_data_loader, test_data_loader,
                    epochs=epochs, optimizer=optimizer_choice,
                    criterion=criterion_choice, patience=10,
                    min_delta=0.0001, disable_wandb=True, verbose=False
                )
                model.eval()

                # Evaluate against MCTS data agent
                print(f"Evaluating {curriculum_type} model ({percentage*100}% data) against MCTS_data_agent...")
                results = evaluate_agents(
                    model, 'mcts_data_agent',
                    games=mcts_eval_games,
                    agent1_criterion=criterion_choice,
                    agent2_criterion=None,
                    mcts_data=preloaded_mcts_data
                )

                a1_win = results["agent1_wins"]
                a2_win = results["agent2_wins"]
                print(a1_win, a2_win)
                win_rate = a1_win / (a1_win + a2_win)
                data_portion_results[curriculum_type][percentage].append(win_rate)
                print(f"Win rate for {curriculum_type} with {percentage*100}% training data: {win_rate:.4f}")

                # Save the model
                model_portion_name = f"model_{curriculum_type.replace('_to_', '_').replace('random', 'random_curriculum')}_portion_{int(percentage*100)}"
                torch.save(model.state_dict(), os.path.join(portion_save_dir, f"{model_portion_name}.pth"))

    # Calculate average win rates and plot
    print("\n--- Plotting Data Portion Results ---")
    plot_save_dir = os.path.join("results", "data_portion_plots")
    os.makedirs(plot_save_dir, exist_ok=True)

    for curriculum_type, percentages_data in data_portion_results.items():
        avg_win_rates = []
        std_dev_win_rates = []
        for percentage in data_percentages:
            rates = percentages_data[percentage]
            avg_win_rates.append(np.mean(rates))
            std_dev_win_rates.append(np.std(rates))

        plt.figure(figsize=(10, 6))
        plt.errorbar(data_percentages, avg_win_rates, yerr=std_dev_win_rates, fmt='-o', capsize=5)
        plt.title(f'Win Rate Against MCTS Data Agent for {curriculum_type} Curriculum')
        plt.xlabel('Percentage of Data Used for Training')
        plt.ylabel('Average Win Rate')
        plt.ylim(0, 1)
        plt.grid(True)
        plot_filename = os.path.join(plot_save_dir, f'{curriculum_type}_win_rate_vs_data_portion.png')
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

    print("\nData portion experiment complete and plots generated.")
    # --- New Experiment: Random Perturbation of Model Weights ---
    print("\n--- Running Model Perturbation Experiment ---")
    perturbation_strength = 0.01 # Define the strength of the perturbation
    perturbation_results = {curriculum_type: [] for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]}

    def perturb_model_weights(model, strength):
        """Applies random perturbation to model weights."""
        perturbed_model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
        perturbed_model.load_state_dict(model.state_dict()) # Start with original weights
        with torch.no_grad():
            for param in perturbed_model.parameters():
                param.add_(torch.randn(param.size()) * strength)
        return perturbed_model

    # Load original models once for each curriculum type (from the first seed's results)
    print("\n--- Loading base models for perturbation experiment (from first seed) ---")
    base_models = {}
    base_seed_for_loading = seeds[0] # Use the first seed's trained models as base
    base_model_dir = os.path.join("results", f'seed_{base_seed_for_loading}_{optimizer_choice}_{criterion_choice}_epoch_{epochs}')
    for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
        model_name = f"model_{curriculum_type.replace('_to_', '_').replace('random', 'random_curriculum')}"
        model_path = os.path.join(base_model_dir, f"{model_name}.pth")
        
        model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        base_models[curriculum_type] = model
    print("Base models loaded.")

    for seed in seeds: # This seed now controls the randomness of the perturbation
        print(f"\n--- Model Perturbation Experiment with perturbation seed: {seed} ---")
        set_seed(seed) # Set seed for the perturbation generation

        perturb_save_dir = os.path.join("results", f'perturbation_seed_{seed}_{optimizer_choice}_{criterion_choice}_epoch_{epochs}')
        os.makedirs(perturb_save_dir, exist_ok=True)

        for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
            original_model = base_models[curriculum_type]

            print(f"Perturbing and evaluating {curriculum_type} model with seed {seed}...")
            perturbed_model = perturb_model_weights(original_model, perturbation_strength)
            
            results = evaluate_agents(
                perturbed_model, 'mcts_data_agent',
                games=mcts_eval_games,
                agent1_criterion=criterion_choice,
                agent2_criterion=None,
                mcts_data=preloaded_mcts_data
            )

            a1_win = results["agent1_wins"]
            a2_win = results["agent2_wins"]
            win_rate = a1_win / (a1_win + a2_win) if (a1_win + a2_win) > 0 else 0
            perturbation_results[curriculum_type].append(win_rate)
            print(f"Win rate for perturbed {curriculum_type} model (perturbation seed {seed}): {win_rate:.4f}")

            # Optionally save the perturbed model (if needed for debugging/analysis)
            # model_name = f"model_{curriculum_type.replace('_to_', '_').replace('random', 'random_curriculum')}"
            # torch.save(perturbed_model.state_dict(), os.path.join(perturb_save_dir, f"{model_name}_perturbed_seed_{seed}.pth"))

    # Calculate average win rates and plot for perturbation experiment
    print("\n--- Plotting Perturbation Results ---")
    perturb_plot_save_dir = os.path.join("results", "perturbation_plots")
    os.makedirs(perturb_plot_save_dir, exist_ok=True)

    avg_perturb_win_rates = []
    std_perturb_win_rates = []
    curriculum_labels = ["easy_to_hard", "hard_to_easy", "random"]

    for ct in curriculum_labels:
        rates = perturbation_results[ct]
        avg_perturb_win_rates.append(np.mean(rates))
        std_perturb_win_rates.append(np.std(rates))

    x = np.arange(len(curriculum_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects = ax.bar(x, avg_perturb_win_rates, width, yerr=std_perturb_win_rates, capsize=5)

    ax.set_ylabel('Average Win Rate Against MCTS Data Agent')
    ax.set_title(f'Win Rate of Perturbed Models (Strength: {perturbation_strength})')
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace('_', ' ').title() for label in curriculum_labels]) # Nicer labels
    ax.set_ylim(0, 1)
    ax.grid(axis='y')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects)

    plot_filename = os.path.join(perturb_plot_save_dir, f'perturbed_models_win_rate_strength_{perturbation_strength}.png')
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")
    plt.close()

    print("\nModel perturbation experiment complete and plots generated.")
