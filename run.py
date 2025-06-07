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
from src.train import train_model
from tqdm import tqdm


if __name__ == "__main__":
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
    parser.add_argument("--criterion", type=str, choices=["mse", "cross_entropy"], default="mse", help="Loss function to use for training.")
    args = parser.parse_args()
    
    epoch = args.epochs
    save_dir = args.save_dir
    optimizer_choice = args.optimizer
    criterion_choice = args.criterion
    save_dir = os.path.join(save_dir, f'{optimizer_choice}_{criterion_choice}_epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    print("Loading easy-to-hard dataset...")
    easy_loader = load_dataset("easy_to_hard")
    print("Training easy-to-hard model...")
    model_easy = TicTacToeCNN()
    train_model(model_easy, easy_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice)
    torch.save(model_easy.state_dict(), os.path.join(save_dir, "model_easy.pth"))

    print("Loading hard-to-easy dataset...")
    hard_loader = load_dataset("hard_to_easy")
    print("Training hard-to-easy model...")
    model_hard = TicTacToeCNN()
    train_model(model_hard, hard_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice)
    torch.save(model_hard.state_dict(), os.path.join(save_dir, "model_hard.pth"))

    print("Loading random dataset...")
    random_loader = load_dataset("random")
    print("Training random model...")
    model_random = TicTacToeCNN()
    train_model(model_random, random_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice)
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