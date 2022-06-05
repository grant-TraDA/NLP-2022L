import numpy as np
import torch


class PositionEncoding(torch.nn.Module):
    def __init__(self, n):
        super(PositionEncoding, self).__init__()
        
        self.E = np.zeros((2**(n-1), n), dtype=np.float32)
        for i in range(n):
            self.E[:, i] = self.periodic(np.arange(2**(n-1)), 2**i)
        
        self.E = torch.tensor(self.E).unsqueeze(0)
        
    def forward(self, X):
        return torch.cat((self.E[:,0:X.shape[1],:].repeat(X.shape[0],1,1), X), dim=2)

    def _apply(self, fn):
        super(PositionEncoding, self)._apply(fn)
        self.E = fn(self.E)
        return self
    
    @staticmethod
    def periodic(x, n):
        return 2*np.abs(np.mod(x/n, 2) - 1) - 1
