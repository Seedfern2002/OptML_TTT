import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeCNN(nn.Module):
    def __init__(self):
        super(TicTacToeCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))  # Probability output for each cell
        return x.view(-1, 3, 3)
    
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1).view(-1, 3, 3)
