from torch import optim, nn
import torch
from tqdm import tqdm
import wandb


def train_model(model, dataloader, epochs=1, optimizer="adam", criterion="mse", momentum=True, disable_wandb=False, verbose=True):
    wandb.init(project="tictactoe", config={
        "epochs": epochs,
        "optimizer": optimizer,
        "criterion": criterion,
        "momentum": momentum
    }, name=f"{optimizer}_{criterion}_epoch_{epochs}",
    mode="disabled" if disable_wandb else "online"
    )
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer == "sgd":
        if momentum:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    else:
        raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")

    if criterion == "mse":
        loss_fn = nn.MSELoss()
    elif criterion == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif criterion == "kl_div":
        loss_fn = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError("Unsupported criterion. Use 'mse' or 'cross_entropy'.")

    for epoch in tqdm(range(epochs), disable=not verbose):
        total_loss = 0
        for x, y in dataloader:
            y = y.float()
            optimizer.zero_grad()
            y = y.view(-1, 9)
            if criterion == "kl_div":
                eps = 1e-8
                y = y + eps
                y = y / y.sum(dim=1, keepdim=True)

            pred = model(x).view(-1, 9)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader):.4f}")
        wandb.log({"epoch": epoch + 1, "loss": total_loss / len(dataloader)})
    wandb.finish()


def train_model_with_early_stopping(model, train_dataloader, test_dataloader, epochs=1, optimizer="adam", criterion="mse", momentum=True, disable_wandb=False, patience=5, min_delta=0.0001, verbose=True):
    wandb.init(project="tictactoe", config={
        "epochs": epochs,
        "optimizer": optimizer,
        "criterion": criterion,
        "momentum": momentum,
        "patience": patience,
        "min_delta": min_delta
    }, name=f"{optimizer}_{criterion}_epoch_{epochs}_early_stopping", 
    mode="disabled" if disable_wandb else "online"
    )
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer == "sgd":
        if momentum:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    else:
        raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")
    
    if criterion == "mse":
        loss_fn = nn.MSELoss()
    elif criterion == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif criterion == "kl_div":
        loss_fn = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError("Unsupported criterion. Use 'mse' or 'cross_entropy'.")

    best_test_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs), disable=not verbose):
        total_train_loss = 0
        model.train()
        for x, y in train_dataloader:
            y = y.float()
            optimizer.zero_grad()

            y = y.view(-1, 9)
            if criterion == "kl_div":
                eps = 1e-8
                y = y + eps
                y = y / y.sum(dim=1, keepdim=True)

            pred = model(x).view(-1, 9)

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        if verbose:
            print(f"Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}")

        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for x_test, y_test in test_dataloader:
                y_test = y_test.float().view(-1, 9)
                pred_test = model(x_test).view(-1, 9)
                loss_test = loss_fn(pred_test, y_test)
                total_test_loss += loss_test.item()

        avg_test_loss = total_test_loss / len(test_dataloader)
        if verbose:
            print(f"Epoch {epoch+1}, Average Test Loss: {avg_test_loss:.4f}")

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "test_loss": avg_test_loss})

        # Early stopping check
        if avg_test_loss < best_test_loss - min_delta:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in test loss.")
                break
    wandb.finish()

