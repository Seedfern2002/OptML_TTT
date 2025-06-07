import numpy as np
import torch
import random
from model import TicTacToeCNN


def evaluate_models(model1, model2, games=1000):
    from tictactoe import TicTacToe

    def select_move(model, game):
        board = np.zeros((2, 3, 3))
        for i, v in enumerate(game.board):
            if v == 'X':
                board[0, i // 3, i % 3] = 1
            elif v == 'O':
                board[1, i // 3, i % 3] = 1
        with torch.no_grad():
            probs = model(torch.tensor([board], dtype=torch.float32)).squeeze().numpy()
        legal = game.available_moves()
        probs = np.array([probs[i//3][i % 3] if i in legal else 0 for i in range(9)])
        s = probs.sum()
        if s == 0:
            return random.choice(legal)
        probs = probs / s
        # print(f'sum of probs: {sum(probs)}')
        # use greedy policy
        # return the largest probability move
        # return np.argmax(probs)
        return np.random.choice(range(9), p=probs)

    results = {"model1": 0, "model2": 0, "draw": 0}
    for i in range(games):
        game = TicTacToe()
        players = [model1, model2] if i % 2 == 0 else [model2, model1]
        while game.winner() is None:
            move = select_move(players[0 if game.current_player == 'X' else 1], game)
            game.make_move(move)
        result = game.winner()
        if result == 'X':
            results["model1" if i % 2 == 0 else "model2"] += 1
        elif result == 'O':
            results["model2" if i % 2 == 0 else "model1"] += 1
        else:
            results["draw"] += 1
    print(results)


if __name__ == "__main__":
    epoch = 10
    print("Loading easy-to-hard dataset...")
    easy_loader = load_dataset("easy_to_hard")
    print("Training easy-to-hard model...")
    model_easy = TicTacToeCNN()
    train_model(model_easy, easy_loader, epochs=epoch)
    torch.save(model_easy.state_dict(), "model_easy.pth")

    print("Loading hard-to-easy dataset...")
    hard_loader = load_dataset("hard_to_easy")
    print("Training hard-to-easy model...")
    model_hard = TicTacToeCNN()
    train_model(model_hard, hard_loader, epochs=epoch)
    torch.save(model_hard.state_dict(), "model_hard.pth")

    print("Loading random dataset...")
    random_loader = load_dataset("random")
    print("Training random model...")
    model_random = TicTacToeCNN()
    train_model(model_random, random_loader, epochs=epoch)
    torch.save(model_random.state_dict(), "model_random.pth")

    print("Evaluating models...")
    print("Evaluating easy model against hard model...")
    evaluate_models(model_easy, model_hard)
    print("Evaluating easy model against random model...")
    evaluate_models(model_easy, model_random)
    print("Evaluating hard model against random model...")
    evaluate_models(model_hard, model_random)

    # model_easy = TicTacToeCNN()
    # model_easy.load_state_dict(torch.load("model_easy.pth"))
    # model_easy.eval()

    # model_hard = TicTacToeCNN()
    # model_hard.load_state_dict(torch.load("model_hard.pth"))
    # model_hard.eval()

    # print with input of all 0
    # prob_easy = model_easy(torch.zeros((1, 2, 3, 3), dtype=torch.float32))
    # prob_hard = model_hard(torch.zeros((1, 2, 3, 3), dtype=torch.float32))
    # print("Normalized probabilities for model_easy:", prob_easy.detach().numpy().flatten() / sum(prob_easy.detach().numpy().flatten()))
    # print("Normalized probabilities for model_hard:", prob_hard.detach().numpy().flatten() / sum(prob_hard.detach().numpy().flatten()))
