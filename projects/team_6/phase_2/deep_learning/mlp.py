from torch import nn


class MLP(nn.Module):
    def __init__(self, start):
        super().__init__()

        layers = [nn.Linear(start, 200),
                  nn.Dropout(0.2),
                  nn.ReLU(),
                  nn.Linear(200, 50),
                  nn.Dropout(0.2),
                  nn.ReLU(),
                  nn.Linear(50, 3)]

        self.net = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        x = self.softmax(x)
        return x
