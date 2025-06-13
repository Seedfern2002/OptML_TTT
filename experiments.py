import os
import random
import pickle
import numpy as np
import torch
import pandas as pd

from model.model import TicTacToeCNN
from src.dataloader import load_dataset
from src.eval import evaluate_agents
from src.train import train_model, train_model_with_early_stopping, train_model_with_test
from utils.eval_utils import eval_models_all_epochs, get_highest_probability


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


def run_curriculum_experiment(seed, preloaded_mcts_data, epochs, optimizer_choice, criterion_choice, default_eval_games, mcts_eval_games):
    """Trains curriculum models and evaluates them against each other, MCTS agent, and random agent."""
    print(f"\n--- Running Curriculum Experiment with seed: {seed} ---")
    set_seed(seed)
    models = {}
    results_summary = {}
    for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
        # Train model for given curriculum type
        train_data_loader = load_dataset(curriculum_type)
        model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
        train_model(model, train_data_loader, epochs=epochs, optimizer=optimizer_choice, 
                    criterion=criterion_choice, disable_wandb=True, verbose=False)
        model.eval()
        models[curriculum_type] = model

    # Evaluate comparisons among curriculum models
    curriculum_types = list(models.keys())
    for i in range(len(curriculum_types)):
        for j in range(i + 1, len(curriculum_types)):
            m1 = models[curriculum_types[i]]
            m2 = models[curriculum_types[j]]
            results = evaluate_agents(m1, m2,
                                      games=default_eval_games,
                                      agent1_criterion=criterion_choice,
                                      agent2_criterion=criterion_choice)
            comp_name = f"{curriculum_types[i]} vs {curriculum_types[j]}"
            results_summary[comp_name] = results

    # Evaluate each model vs pre-computed MCTS agent
    for curriculum_type, model in models.items():
        results = evaluate_agents(model, 'mcts_data_agent',
                                  games=mcts_eval_games,
                                  agent1_criterion=criterion_choice,
                                  agent2_criterion=None,
                                  mcts_data=preloaded_mcts_data)
        comp_name = f"{curriculum_type} vs MCTS_data_agent"
        results_summary[comp_name] = results

    # Evaluate each model vs pure random agent
    for curriculum_type, model in models.items():
        results = evaluate_agents(model, 'random_agent',
                                  games=default_eval_games,
                                  agent1_criterion=criterion_choice,
                                  agent2_criterion=None)
        comp_name = f"{curriculum_type} vs Random_agent"
        results_summary[comp_name] = results
    
    return models, results_summary


def run_data_portion_experiment(seed, preloaded_mcts_data, epochs, optimizer_choice, criterion_choice, mcts_eval_games):
    """Trains models on varying portions of data and evaluates against the MCTS agent."""
    print(f"\n--- Running Data Portion Experiment with seed: {seed} ---")
    set_seed(seed)
    data_percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
    data_portion_results = {ct: {p: [] for p in data_percentages} for ct in ["easy_to_hard", "hard_to_easy", "random"]}
    for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
        test_data_loader = load_dataset(curriculum_type, split='test', data_percentage=1.0)
        for percentage in data_percentages:
            train_data_loader = load_dataset(curriculum_type, split='train', data_percentage=percentage)
            model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
            train_model_with_early_stopping(model, train_data_loader, test_data_loader,
                                            epochs=epochs, optimizer=optimizer_choice,
                                            criterion=criterion_choice, patience=10,
                                            min_delta=0.0001, disable_wandb=True, verbose=False)
            model.eval()
            results = evaluate_agents(model, 'mcts_data_agent',
                                      games=mcts_eval_games,
                                      agent1_criterion=criterion_choice,
                                      agent2_criterion=None,
                                      mcts_data=preloaded_mcts_data)
            a1_win = results["agent1_wins"]
            a2_win = results["agent2_wins"]
            win_rate = a1_win / (a1_win + a2_win) if (a1_win + a2_win) > 0 else 0
            data_portion_results[curriculum_type][percentage].append(win_rate)
    return data_portion_results


def load_base_models(optimizer_choice, criterion_choice, epochs):
    """Trains and returns base models for each curriculum type using a fixed seed."""
    set_seed(42)
    base_models = {}
    for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
        train_data_loader = load_dataset(curriculum_type)
        model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
        train_model(model, train_data_loader, epochs=epochs, optimizer=optimizer_choice, 
                    criterion=criterion_choice, disable_wandb=True, verbose=False)
        model.eval()
        base_models[curriculum_type] = model
    return base_models


def perturb_model_weights(model, strength, criterion_choice):
    """Applies random perturbation to model weights."""
    perturbed_model = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    perturbed_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        for param in perturbed_model.parameters():
            param.add_(torch.randn(param.size()) * strength)
    return perturbed_model


def run_perturbation_experiment(seed, base_models, preloaded_mcts_data, mcts_eval_games, criterion_choice, perturbation_strength):
    """Perturbs each base model and evaluates against the MCTS agent."""
    print(f"\n--- Running Perturbation Experiment with perturbation seed: {seed} ---")
    set_seed(seed)
    perturbation_results = {ct: [] for ct in ["easy_to_hard", "hard_to_easy", "random"]}
    for curriculum_type in ["easy_to_hard", "hard_to_easy", "random"]:
        original_model = base_models[curriculum_type]
        perturbed_model = perturb_model_weights(original_model, perturbation_strength, criterion_choice)
        results = evaluate_agents(perturbed_model, 'mcts_data_agent',
                                  games=mcts_eval_games,
                                  agent1_criterion=criterion_choice,
                                  agent2_criterion=None,
                                  mcts_data=preloaded_mcts_data)
        a1_win = results["agent1_wins"]
        a2_win = results["agent2_wins"]
        win_rate = a1_win / (a1_win + a2_win) if (a1_win + a2_win) > 0 else 0
        perturbation_results[curriculum_type].append(win_rate)
    return perturbation_results


def run_all_curriculum_experiments(seeds, preloaded_mcts_data, epochs, optimizer_choice, criterion_choice, default_eval_games, mcts_eval_games):
    """Runs curriculum experiments for all seeds and aggregates results into a pandas DataFrame.
    Each row represents a matchup between two agents with the seed, agent1 wins, agent2 wins, and draws.
    """
    all_rows = []
    for seed in seeds:
        _, results_summary = run_curriculum_experiment(seed, preloaded_mcts_data, epochs, optimizer_choice, 
                                                       criterion_choice, default_eval_games, mcts_eval_games)
        for matchup, res in results_summary.items():
            row = {
                "seed": seed,
                "matchup": matchup,
                "agent1_wins": res.get("agent1_wins", 0),
                "agent2_wins": res.get("agent2_wins", 0),
                "draws": res.get("draws", 0)
            }
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    return df


def run_training_experiments(epoch, save_dir, optimizer_choice, criterion_choice, momentum_choice, disable_wandb, save_per_epoch, no_momentum, with_test, log_file):
    """Train and evaluate models for all curriculum types and save results."""
    set_seed(42)
    save_dir = os.path.join(save_dir, f'{optimizer_choice}_{criterion_choice}_epoch_{epoch}')
    save_dir += '_no_momentum' if no_momentum else ''
    save_dir += '_with_test' if with_test else ''
    os.makedirs(save_dir, exist_ok=True)
    if log_file:
        easy_to_hard_log_file = os.path.join(save_dir, "easy_to_hard_training.log")
        hard_to_easy_log_file = os.path.join(save_dir, "hard_to_easy_training.log")
        random_log_file = os.path.join(save_dir, "random_training.log")
        open(easy_to_hard_log_file, 'w').close()
        open(hard_to_easy_log_file, 'w').close()
        open(random_log_file, 'w').close()

    print("Training easy-to-hard model...")
    model_easy = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    model_save_dir = os.path.join(save_dir, "model_easy2hard")
    os.makedirs(model_save_dir, exist_ok=True)
    if with_test:
        easy_loader = load_dataset("easy_to_hard", split='train')
        easy_test_loader = load_dataset("easy_to_hard", split='test')
        train_model_with_test(model_easy, easy_loader, easy_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=easy_to_hard_log_file if log_file else None)
    else:
        easy_loader = load_dataset("easy_to_hard")
        train_model(model_easy, easy_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=easy_to_hard_log_file if log_file else None, save_per_epoch=save_per_epoch, save_dir=model_save_dir)
    torch.save(model_easy.state_dict(), os.path.join(save_dir, "model_easy.pth"))

    print("Training hard-to-easy model...")
    model_hard = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    model_save_dir = os.path.join(save_dir, "model_hard2easy")
    os.makedirs(model_save_dir, exist_ok=True)
    if with_test:
        hard_loader = load_dataset("hard_to_easy", split='train')
        hard_test_loader = load_dataset("hard_to_easy", split='test')
        train_model_with_test(model_hard, hard_loader, hard_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=hard_to_easy_log_file if log_file else None)
    else:
        hard_loader = load_dataset("hard_to_easy")
        train_model(model_hard, hard_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=hard_to_easy_log_file if log_file else None, save_per_epoch=save_per_epoch, save_dir=model_save_dir)
    torch.save(model_hard.state_dict(), os.path.join(save_dir, "model_hard.pth"))

    print("Training random model...")
    model_random = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    model_save_dir = os.path.join(save_dir, "model_random")
    os.makedirs(model_save_dir, exist_ok=True)
    if with_test:
        random_loader = load_dataset("random", split='train')
        random_test_loader = load_dataset("random", split='test')
        train_model_with_test(model_random, random_loader, random_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=random_log_file if log_file else None)
    else:
        random_loader = load_dataset("random")
        train_model(model_random, random_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=random_log_file if log_file else None, save_per_epoch=save_per_epoch, save_dir=model_save_dir)
    torch.save(model_random.state_dict(), os.path.join(save_dir, "model_random.pth"))

    print("Evaluating models...")
    model_name = f'{optimizer_choice}_{criterion_choice}_epoch_{epoch}'
    model_name += '_no_momentum' if no_momentum else ''
    model_name += '_with_test' if with_test else ''
    results_easy2hard = eval_models_all_epochs(os.path.join('results', model_name), "easy2hard", "hard2easy", per_epochs=5)
    results_hard2easy = eval_models_all_epochs(os.path.join('results', model_name), "hard2easy", "random", per_epochs=5)
    results_random = eval_models_all_epochs(os.path.join('results', model_name), "easy2hard", "random", per_epochs=5)
    with open(os.path.join('results', model_name, "comparison_results.pkl"), 'wb') as f:
        pickle.dump({
            "easy2hard_vs_hard2easy": results_easy2hard,
            "hard2easy_vs_random": results_hard2easy,
            "easy2hard_vs_random": results_random
        }, f)


def run_all_training_experiments():
    """Run all main training experiments with various settings."""
    epochs = 50
    save_dir = 'results'
    no_momentum = False
    disable_wandb = True
    save_per_epoch = True
    run_training_experiments(epochs, save_dir, "adam", "mse", not no_momentum, disable_wandb, save_per_epoch, no_momentum, with_test=False, log_file=True)
    run_training_experiments(epochs, save_dir, "adam", "kl_div", not no_momentum, disable_wandb, save_per_epoch, no_momentum, with_test=False, log_file=True)
    run_training_experiments(epochs, save_dir, "sgd", "mse", not no_momentum, disable_wandb, save_per_epoch, no_momentum, with_test=False, log_file=True)
    run_training_experiments(epochs, save_dir, "adam", "mse", not no_momentum, disable_wandb, save_per_epoch, no_momentum, with_test=True, log_file=True)
    run_training_experiments(epochs, save_dir, "adam", "kl_div", not no_momentum, disable_wandb, save_per_epoch, no_momentum, with_test=True, log_file=True)
    run_training_experiments(epochs, save_dir, "sgd", "mse", not no_momentum, disable_wandb, save_per_epoch, no_momentum, with_test=True, log_file=True)


def get_models(model_name):
    """Load trained models for each curriculum type from disk."""
    model_easy = TicTacToeCNN(kl_div=False)
    model_easy.load_state_dict(torch.load(os.path.join('results', model_name, "model_easy.pth")))
    model_hard = TicTacToeCNN(kl_div=False)
    model_hard.load_state_dict(torch.load(os.path.join('results', model_name, "model_hard.pth")))
    model_random = TicTacToeCNN(kl_div=False)
    model_random.load_state_dict(torch.load(os.path.join('results', model_name, "model_random.pth")))
    return [model_easy.eval(), model_hard.eval(), model_random.eval()], ['easy2hard', 'hard2easy', 'random']


def run_highest_probability_experiment(model_name, preloaded_mcts_data, save_dir='results', kl_div=False):
    """Compute and save the highest output probability for each model and MCTS agent."""
    models, model_names = get_models(model_name)
    prob_models, prob_mcts = get_highest_probability(models, preloaded_mcts_data, kl_div=kl_div)
    save_path = os.path.join(save_dir, f'{model_name}', "highest_probabilities.txt")
    with open(save_path, 'w') as f:
        for key, prob in prob_models.items():
            line = f"{model_names[key]}: {np.mean(prob):.4f} ± {np.std(prob):.4f}\n"
            f.write(line)
            print(line.strip())
        line = f"MCTS data agent: {np.mean(prob_mcts):.4f} ± {np.std(prob_mcts):.4f}\n"
        f.write(line)
        print(line.strip())
    print(f"Results saved at: {save_path}")


def run_random_vs_mcts_experiment(seeds, preloaded_mcts_data, criterion_choice, default_eval_games):
    """Evaluate random agent vs MCTS data agent for multiple seeds."""
    random_vs_mcts_rates = {}
    for seed in seeds:
        set_seed(seed)
        results = evaluate_agents("random_agent", "mcts_data_agent",
                                games=default_eval_games,
                                agent1_criterion=criterion_choice,
                                agent2_criterion=None,
                                mcts_data=preloaded_mcts_data)
        wins = results.get("agent1_wins", 0)
        losses = results.get("agent2_wins", 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        random_vs_mcts_rates[seed] = win_rate
        print(f"Seed {seed}: Random Agent vs MCTS Data Agent Win Rate: {win_rate:.4f}")
    rates = list(random_vs_mcts_rates.values())
    print(f"Random Agent vs MCTS Data Agent Average Win Rate: {np.mean(rates):.4f} ± {np.std(rates):.4f}")