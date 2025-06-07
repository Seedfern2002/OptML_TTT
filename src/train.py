from torch import optim, nn
import tqdm


def train_model(model, dataloader, epochs=1):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    # use kl divergence loss for probabilities
    # loss_fn = nn.KLDivLoss()
    # TODO: try different loss func later
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
