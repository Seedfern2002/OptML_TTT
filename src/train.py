from torch import optim, nn
import torch
from tqdm import tqdm
import wandb


def train_model(model, dataloader, epochs=1, optimizer="adam", criterion="mse", momentum=True):
    wandb.init(project="tictactoe", config={
        "epochs": epochs,
        "optimizer": optimizer,
        "criterion": criterion,
        "momentum": momentum
    }, name=f"{optimizer}_{criterion}_epoch_{epochs}"
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
   
    for epoch in tqdm(range(epochs)):
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
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": total_loss})
    wandb.finish()


def train_model_with_test(model, train_dataloader, test_dataloader, epochs=1, optimizer="adam", criterion="mse", momentum=True):
    wandb.init(project="tictactoe", config={
        "epochs": epochs,
        "optimizer": optimizer,
        "criterion": criterion,
        "momentum": momentum
    }, name=f"{optimizer}_{criterion}_epoch_{epochs}"
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
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0
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
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_test, y_test in test_dataloader:
                y_test = y_test.float().view(-1, 9)
                pred_test = model(x_test).view(-1, 9)
                loss_test = loss_fn(pred_test, y_test)
                test_loss += loss_test.item()
        
        print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")

        wandb.log({"epoch": epoch + 1, "train_loss": total_loss, "test_loss": test_loss})
    wandb.finish()
