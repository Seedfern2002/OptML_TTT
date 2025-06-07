from torch import optim, nn
from tqdm import tqdm


def train_model(model, dataloader, epochs=1, optimizer="adam", criterion="mse"):
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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
