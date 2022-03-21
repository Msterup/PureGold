import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=378, out_channels=378):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(in_channels, out_channels)

        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        residual = x
        out = self.fc1(x)

        out = self.relu(out)

        out = self.fc2(out)

        out = self.relu(out)

        out = self.relu(out)
        out = self.fc3(out)

        out = self.relu(out)
        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=378, out_channels=378):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(378, 1000)  # 5*5 from image dimension
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)
        self.relu = nn.ReLU()


    def make_layer(self, block, blocks):
        layers = []
        for i in range(1, blocks):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out