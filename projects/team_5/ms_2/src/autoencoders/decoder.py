from autoencoders.position_encoding import PositionEncoding

import torch


class Decoder(torch.nn.Module):
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
                 max_log2len=8):
        """
            Decoder(
              # Input/Output params
              word_embedding_size,
              document_embedding_size,

              # Transformer params
              d_model,
              nhead,
              num_layers,
              dropout=0.1
              
              # Pos. Enc. params
              max_log2len=8):
            )

        Decoding module. It has three parts: 
        * input linear layer
        * inner transformer encoder
        * output linear layers

        As inner transformer encoder restricts dimensionality to be devisible,
        the input linear layer allows for any dimensionality inputs. Likewise,
        the output linear layers, serve the same function, they also create 
        expectation representation and logarithm of variance representation.

        """
        super(Decoder, self).__init__()
        
        # Parameters
        self.word_embedding_size     = word_embedding_size
        self.document_embedding_size = document_embedding_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Positional Encoding
        self.position_encoding = PositionEncoding(max_log2len) # Concatentates position encoding.

        # Input Layer: 
        self.input_layer = torch.nn.Linear(
            document_embedding_size + max_log2len, # We firstly concatenate the position encoding.
            d_model
        )

        # Transformer Block
        self.inner_layer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout),
            num_layers=num_layers
        )

        # Transformer Block
        self.returning = torch.nn.Linear(d_model, word_embedding_size)
        
    def forward(self, X):
        
        X = self.position_encoding(X)
        X = torch.relu(self.input_layer(X))
        X = torch.relu(self.inner_layer(X))
        return self.returning(X)
