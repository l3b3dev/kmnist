from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    conv1 (channels = 10, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    conv2 (channels = 50, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    Linear (256) -> Relu -> Linear (10) -> LogSoftmax

    Need to reshape outputs from the last conv layer prior to feeding them into
    the linear layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 50, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # original size 1* 28 * 28
        x = F.relu(self.conv1(x))  # 10 * 24 * 24
        x = self.pool(x)  # 10 * 12 * 12
        x = F.relu(self.conv2(x))  # 50 * 8 * 8
        x = self.pool(x)  # 50 * 4 * 4 = 800
        x = x.view(x.shape[0], -1)  # flattening the inputs.
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
