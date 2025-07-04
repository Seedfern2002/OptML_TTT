import numpy as np
import torch
import random
from model.model import TicTacToeCNN
from src.dataloader import load_dataset
import argparse
import os
from utils.eval_utils import evaluate_models, eval_models_all_epochs
from src.train import train_model, train_model_with_test
import pickle


if __name__ == "__main__":
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
    parser.add_argument("--disable_wandb", action='store_true', dest='disable_wandb', help="Disable Weights & Biases logging.")
    parser.add_argument("--log_file", action='store_true', dest='log_file', help="Whether to log training progress to a file.")
    parser.add_argument("--save_per_epoch", action='store_true', dest='save_per_epoch', help="Whether to save the model after each epoch.")
    args = parser.parse_args()
    
    epoch = args.epochs
    save_dir = args.save_dir
    optimizer_choice = args.optimizer
    criterion_choice = args.criterion
    momentum_choice = not args.no_momentum
    disable_wandb = args.disable_wandb
    save_per_epoch = args.save_per_epoch
    
    print(f"Training with {optimizer_choice} optimizer, {criterion_choice} criterion, momentum={momentum_choice}, for {epoch} epochs.")
    save_dir = os.path.join(save_dir, f'{optimizer_choice}_{criterion_choice}_epoch_{epoch}')
    save_dir += '_no_momentum' if args.no_momentum else ''
    save_dir += '_with_test' if args.with_test else ''
    os.makedirs(save_dir, exist_ok=True)

    if args.log_file:
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
    if args.with_test:
        easy_loader = load_dataset("easy_to_hard", split='train')
        easy_test_loader = load_dataset("easy_to_hard", split='test')
        train_model_with_test(model_easy, easy_loader, easy_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=easy_to_hard_log_file if args.log_file else None)
    else:
        easy_loader = load_dataset("easy_to_hard")
        train_model(model_easy, easy_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=easy_to_hard_log_file if args.log_file else None, save_per_epoch=save_per_epoch, save_dir=model_save_dir)
    torch.save(model_easy.state_dict(), os.path.join(save_dir, "model_easy.pth"))

    print("Training hard-to-easy model...")
    model_hard = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    model_save_dir = os.path.join(save_dir, "model_hard2easy")
    os.makedirs(model_save_dir, exist_ok=True)
    if args.with_test:
        hard_loader = load_dataset("hard_to_easy", split='train')
        hard_test_loader = load_dataset("hard_to_easy", split='test')
        train_model_with_test(model_hard, hard_loader, hard_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=hard_to_easy_log_file if args.log_file else None)
    else:
        hard_loader = load_dataset("hard_to_easy")
        train_model(model_hard, hard_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=hard_to_easy_log_file if args.log_file else None, save_per_epoch=save_per_epoch, save_dir=model_save_dir)
    torch.save(model_hard.state_dict(), os.path.join(save_dir, "model_hard.pth"))

    print("Training random model...")
    model_random = TicTacToeCNN(kl_div=(criterion_choice == "kl_div"))
    model_save_dir = os.path.join(save_dir, "model_random")
    os.makedirs(model_save_dir, exist_ok=True)
    if args.with_test:
        random_loader = load_dataset("random", split='train')
        random_test_loader = load_dataset("random", split='test')
        train_model_with_test(model_random, random_loader, random_test_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=random_log_file if args.log_file else None)
    else:
        random_loader = load_dataset("random")
        train_model(model_random, random_loader, epochs=epoch, optimizer=optimizer_choice, criterion=criterion_choice, momentum=momentum_choice, disable_wandb=disable_wandb, log_file=random_log_file if args.log_file else None, save_per_epoch=save_per_epoch, save_dir=model_save_dir)
    torch.save(model_random.state_dict(), os.path.join(save_dir, "model_random.pth"))

    model_name = f'{optimizer_choice}_{criterion_choice}_epoch_{epoch}'
    model_name += '_no_momentum' if args.no_momentum else ''
    model_name += '_with_test' if args.with_test else ''
    print("Evaluating models...")

    results_easy2hard = eval_models_all_epochs(f'results/{model_name}', "easy2hard", "hard2easy", per_epochs=5)
    results_hard2easy = eval_models_all_epochs(f'results/{model_name}', "hard2easy", "random", per_epochs=5)
    results_random = eval_models_all_epochs(f'results/{model_name}', "easy2hard", "random", per_epochs=5)

    with open(f'results/{model_name}/comparison_results.pkl', 'wb') as f:
        pickle.dump({
            "easy2hard_vs_hard2easy": results_easy2hard,
            "hard2easy_vs_random": results_hard2easy,
            "easy2hard_vs_random": results_random
        }, f)