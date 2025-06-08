import numpy as np
import torch
import random
from model.model import TicTacToeCNN
from src.dataloader import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from src.eval import evaluate_models
from src.train import train_model, train_model_with_test
from tqdm import tqdm


if __name__ == "__main__":
    # model_mse = TicTacToeCNN()
    # model_mse.load_state_dict(torch.load('/mnt/aimm/scratch/yecheng/TTT/results/adam_mse_epoch_10/model_easy.pth'))
    # model_mse.eval()

    # model_ce = TicTacToeCNN()
    # model_ce.load_state_dict(torch.load('/mnt/aimm/scratch/yecheng/TTT/results/adam_cross_entropy_epoch_10/model_easy.pth'))
    # model_ce.eval()

    # model_kl = TicTacToeCNN(kl_div=True)
    # model_kl.load_state_dict(torch.load('/mnt/aimm/scratch/yecheng/TTT/results/adam_kl_div_epoch_10/model_easy.pth'))
    # model_kl.eval()

    # results = evaluate_models(model_mse, model_ce, games=1000)
    # print("Results (MSE vs Cross Entropy):", results)

    # results = evaluate_models(model_mse, model_kl, games=1000)
    # print("Results (MSE vs KL Divergence):", results)

    # results = evaluate_models(model_ce, model_kl, games=1000)
    # print("Results (Cross Entropy vs KL Divergence):", results)
    # set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    parser = argparse.ArgumentParser(description="Train and evaluate TicTacToe models.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory where dataset is stored.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam", help="Optimizer to use for training.")
    parser.add_argument("--criterion", type=str, choices=["mse", "cross_entropy", "kl_div"], default="mse", help="Loss function to use for training.")
    parser.add_argument("--no_momentum", action='store_true', dest='no_momentum', help="Use momentum in SGD optimizer.")
    parser.add_argument("--with_test", action='store_true', dest='with_test', help="Whether to use test set for evaluation during training.")
    args = parser.parse_args()
    
    epoch = args.epochs
    save_dir = args.save_dir
    optimizer_choice = args.optimizer
    criterion_choice = args.criterion
    momentum_choice = not args.no_momentum
    print(f"Training with {optimizer_choice} optimizer, {criterion_choice} criterion, momentum={momentum_choice}, for {epoch} epochs.")
    save_dir = os.path.join(save_dir, f'{optimizer_choice}_{criterion_choice}_epoch_{epoch}')
    save_dir += '_no_momentum' if args.no_momentum else ''
    save_dir += '_with_test' if args.with_test else ''
    os.makedirs(save_dir, exist_ok=True)

    print("Training easy-to-hard model...")
    model_easy = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    if args.with_test:
        easy_loader = load_dataset("easy_to_hard", split='train')
        easy_test_loader = load_dataset("easy_to_hard", split='test')
        train_model_with_test(model_easy, easy_loader, easy_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice)
    else:
        easy_loader = load_dataset("easy_to_hard")
        train_model(model_easy, easy_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice)
    torch.save(model_easy.state_dict(), os.path.join(save_dir, "model_easy.pth"))

    print("Training hard-to-easy model...")
    model_hard = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    if args.with_test:
        hard_loader = load_dataset("hard_to_easy", split='train')
        hard_test_loader = load_dataset("hard_to_easy", split='test')
        train_model_with_test(model_hard, hard_loader, hard_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice)
    else:
        hard_loader = load_dataset("hard_to_easy")
        train_model(model_hard, hard_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice)
    torch.save(model_hard.state_dict(), os.path.join(save_dir, "model_hard.pth"))

    print("Training random model...")
    model_random = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    if args.with_test:
        random_loader = load_dataset("random", split='train')
        random_test_loader = load_dataset("random", split='test')
        train_model_with_test(model_random, random_loader, random_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice)
    else:
        random_loader = load_dataset("random")
        train_model(model_random, random_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice)
    torch.save(model_random.state_dict(), os.path.join(save_dir, "model_random.pth"))

    print("Evaluating models...")
    print("Evaluating easy model against hard model...")
    results_evh = evaluate_models(model_easy, model_hard)
    print("Results (easy vs hard):", results_evh)
    print("Evaluating easy model against random model...")
    results_evr = evaluate_models(model_easy, model_random)
    print("Results (easy vs random):", results_evr)
    print("Evaluating hard model against random model...")
    results_hvr = evaluate_models(model_hard, model_random)
    print("Results (hard vs random):", results_hvr)

    # log the results to a file
    with open(os.path.join(save_dir, "evaluation_results.txt"), "w") as f:
        f.write("Results (easy vs hard): " + str(results_evh) + "\n")
        f.write("Results (easy vs random): " + str(results_evr) + "\n")
        f.write("Results (hard vs random): " + str(results_hvr) + "\n")