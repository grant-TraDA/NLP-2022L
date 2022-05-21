from torch import nn

class MLP(nn.Module):
    def __init__(self, dims, dropout = 0.5, activation = None):
        super().__init__()
        # assert len(dims) > 2, 'must have at least 3 dimensions, for dimension in and dimension out'
        activation = activation if activation is not None else nn.ReLU

        layers = []
        pairs = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(pairs):
            is_last = ind >= (len(pairs) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if not is_last:
                layers.append(nn.Dropout(dropout))
                layers.append(activation())

        # layers.append(nn.Softmax())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).flatten(1)