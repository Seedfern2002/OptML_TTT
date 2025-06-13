import matplotlib.pyplot as plt
import os
import pickle
import numpy as np


def load_loss_data(file_path, with_test=False):
    """Load loss values from a log file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file {file_path} does not exist.")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    losses = []
    for line in lines:
        if with_test:
            test_loss = float(line.strip().split('Test Loss:')[1])
            losses.append(test_loss)
        else:
            parts = line.split(':')
            loss = float(parts[1].strip())
            losses.append(loss)

    return losses


def plot_loss(optimizer, loss_function, epochs, with_test=False, no_momentum=False):
    """Plot training and/or test loss curves from log files."""
    save_dir = f"results/{optimizer}_{loss_function}_epoch_{epochs}"
    if no_momentum:
        save_dir += "_no_momentum"
    if with_test:
        save_dir += "_with_test"

    log_files = ['easy_to_hard_training.log', 'hard_to_easy_training.log', 'random_training.log']
    plt.figure(figsize=(10, 6))
    for log_file in log_files:
        file_path = os.path.join(save_dir, log_file)
        losses = load_loss_data(file_path, with_test=with_test)
        print(f"Loaded {len(losses)} losses from {file_path}")
        plt.plot(losses, label=log_file.replace('_', ' ').replace('training.log', '').title())

    # enlarge the font size of the plot, labels, ticks, and legend
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    save_name = f"{optimizer}_{loss_function}_epoch_{epochs}"
    if with_test:
        save_name += "_with_test"
    plt.show()
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.close()


def plot_win_to_loss_ratio(optimizer, loss_function, epochs, with_test=False, no_momentum=False):
    """Plot win/loss ratio curves for model comparisons."""
    save_dir = f"results/{optimizer}_{loss_function}_epoch_{epochs}"
    if no_momentum:
        save_dir += "_no_momentum"
    if with_test:
        save_dir += "_with_test"

    comparison_file = os.path.join(save_dir, "comparison_results.pkl")
    comparison_data = pickle.load(open(comparison_file, 'rb'))
    keys = ["easy2hard_vs_hard2easy", "easy2hard_vs_random", "hard2easy_vs_random"]
    plt.figure(figsize=(10, 6))
    for key in keys:
        model1, model2 = key.split('_vs_')
        model1 = 'model_' + model1
        model2 = 'model_' + model2
        data = comparison_data[key]
        epochs = data.keys()
        win_to_loss = []
        for epoch in epochs:
            win = data[epoch][model1]
            loss = data[epoch][model2]
            win_to_loss.append(win / loss if loss > 0 else float('inf'))
        # plot the win to loss rate, the x-axis is the epoch, the y-axis is the win to loss rate
        plt.plot(list(epochs), win_to_loss, label=key.replace('_', ' ').title())
    
    # enlarge the font size of the plot, labels, ticks, and legend
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Win / Loss Ratio', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    save_name = f"{optimizer}_{loss_function}_win_to_loss_rate"
    if with_test:
        save_name += "_with_test"
    plt.show()
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.close()


def plot_data_portion_results(data_portion_results, data_percentages):
    """Plot win/loss rates as a function of data portion used for training."""
    for curriculum_type, percentage_data in data_portion_results.items():
        avg_win_rates = []
        std_win_rates = []
        for pct in data_percentages:
            rates = percentage_data[pct]
            avg_win_rates.append(np.mean(rates))
            std_win_rates.append(np.std(rates))
        plt.figure(figsize=(10, 6))
        plt.errorbar(data_percentages, avg_win_rates, yerr=std_win_rates, fmt='-o', capsize=5, label=curriculum_type)
        plt.xlabel('Percentage of Data Used for Training', fontsize=20)
        plt.ylabel('Average Win to Loss Rate', fontsize=20)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
        plt.close()


def plot_perturbation_results(perturbation_results, perturbation_strength):
    """Plot win rates against MCTS agent for different perturbation strengths."""
    curriculum_labels = list(perturbation_results.keys())
    avg_win_rates = []
    std_win_rates = []
    for ct in curriculum_labels:
        rates = perturbation_results[ct]
        avg_win_rates.append(np.mean(rates))
        std_win_rates.append(np.std(rates))
    x = np.arange(len(curriculum_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, avg_win_rates, width, yerr=std_win_rates, capsize=5)
    ax.set_ylabel('Average Win Rate Against MCTS Data Agent', fontsize=20)
    ax.set_title(f'Perturbation Win Rate (Strength: {perturbation_strength})', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(curriculum_labels, fontsize=16)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=16)
    plt.show()
    plt.savefig('perturbation.png', bbox_inches='tight')
    plt.close()


def plot_perturbation_results_by_strength(multi_results, strengths):
    """Plot win/loss rates for multiple perturbation strengths and curricula."""
    curriculum_types = list(multi_results.keys())
    means_by_curriculum = {}
    stds_by_curriculum = {}
    for curr in curriculum_types:
        means = []
        stds = []
        for s in strengths:
            rates = multi_results[curr][str(s)]
            means.append(np.mean(rates))
            stds.append(np.std(rates))
        means_by_curriculum[curr] = means
        stds_by_curriculum[curr] = stds
    x = np.arange(len(strengths))
    total_width = 0.8
    bar_width = total_width / len(curriculum_types)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, curr in enumerate(curriculum_types):
        ax.bar(x + i * bar_width, means_by_curriculum[curr], width=bar_width, 
               yerr=stds_by_curriculum[curr], capsize=5, label=curr)
    ax.set_xticks(x + total_width / 2 - bar_width / 2)
    ax.set_xticklabels([str(s) for s in strengths], fontsize=16)
    ax.set_xlabel("Perturbation Strength", fontsize=20)
    ax.set_ylabel("Average Win to Loss Rate", fontsize=20)
    ax.legend(title="Curriculum", fontsize=20, title_fontsize=18)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=16)
    plt.show()
    plt.close()


def plot_all_training_results():
    """Plot all main training and evaluation results."""
    plot_loss("adam", "mse", 50, with_test=False, no_momentum=False)
    plot_loss("adam", "mse", 50, with_test=True, no_momentum=False)
    plot_loss("adam", "kl_div", 50, with_test=False, no_momentum=False)
    plot_loss("sgd", "mse", 50, with_test=False, no_momentum=False)
    plot_win_to_loss_ratio("adam", "mse", 50)
    plot_win_to_loss_ratio("adam", "kl_div", 50)
    plot_win_to_loss_ratio("sgd", "mse", 50)