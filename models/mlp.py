from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear (256) -> ReLU -> Linear(64) -> ReLU -> Linear(10) -> ReLU-> LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)

        return x
