from autoencoders.decoder import Decoder
from autoencoders.encoder import Encoder

import torch


class Autoencoder(torch.nn.Module):
    def __init__(self,
               # Input/Output params
               word_embedding_size,
               document_embedding_size,

               # Transformer params
               d_model,
               nhead,
               num_layers,
               dropout=0.1,

               # Pos. Enc. params
               max_log2len=8,

               # If use separate variance
               variational=False):

        super(Autoencoder, self).__init__()
        self.variational = variational
        self.encoder = Encoder(word_embedding_size, document_embedding_size, d_model, nhead, num_layers, dropout, max_log2len)
        self.decoder = Decoder(word_embedding_size, document_embedding_size, d_model, nhead, num_layers, dropout, max_log2len)
    
    def forward(self, X, return_Z=False):
        n = X.shape[1]
        X = self.encoder(X, keepdim=True, return_variance=self.variational)
        Z = X
        if self.variational and self.training:
            X = X[0] + torch.exp(0.5*X[1])*torch.randn_like(X[1])
        elif self.variational:
            X = X[0]
        X = X.repeat(1, n, 1)
        X = self.decoder(X)
        if return_Z:
            return X, Z
        else:
            return X
