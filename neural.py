import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(378, 378),
            nn.ReLU(),
            nn.Linear(378, 378),
            nn.ReLU(),
            nn.Linear(378, 378),
            nn.ReLU(),
            nn.Linear(378, 1),

                      )

    def forward(self, input):
        return self.online(input)


