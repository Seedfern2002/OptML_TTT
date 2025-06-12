import matplotlib.pyplot as plt
import os
import pickle

def load_loss_data(file_path, with_test=False):
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
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.close()

def plot_win_to_loss_rate(optimizer, loss_function, epochs, with_test=False, no_momentum=False):
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
    plt.ylabel('Win to Loss Rate', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    save_name = f"{optimizer}_{loss_function}_win_to_loss_rate"
    if with_test:
        save_name += "_with_test"
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.close()

    