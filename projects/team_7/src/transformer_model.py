import math
from transformers import GPT2LMHeadModel

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import Transformer

import positional_encoding
import accent_tokenizer


class Gtp2Model(nn.Module):
    def __init__(self):
        super(Gtp2Model, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, gpt2_input):
        return self.model(gpt2_input, labels=gpt2_input)


class PoemModel(nn.Module):
    """
    Model described in Poem Generation Schema.drawio.png
    """

    def __init__(self, ntoken, d_model, nhead, dim_feedforward, nlayers, accent_to_gtp2_matrix, dropout=0.5):
        super(PoemModel, self).__init__()

        self.emb = nn.Embedding(ntoken, d_model)
        self.pos_encoder = positional_encoding.PositionalEncoding(
            d_model, dropout
        )
        self.trans = Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=nlayers,
            num_decoder_layers=nlayers, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.out = nn.Linear(d_model, ntoken)
        self.AccentTokenizer = accent_tokenizer.AccentToGtp2Tokenizer(
            accent_to_gtp2_matrix
        )
        self.d_model = d_model

    def encode(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return x

    def forward(
            self,
            src, tgt,
            src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask,
            memory_key_padding_mask):
        '''
        src - list of accent ids in first line
        tgt - list of accent ids in second line
        '''

        src = self.encode(src)
        tgt = self.encode(tgt)
        out = self.trans(
            src, tgt,
            src_mask, tgt_mask,
            None,
            src_padding_mask, tgt_padding_mask,
            memory_key_padding_mask)
        out = self.out(out)
        return out

    def get_matrix(
            self,
            src, tgt,
            src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask,
            memory_key_padding_mask):
        src = self.encode(src)
        tgt = self.encode(tgt)
        out = self.trans(
            src, tgt,
            src_mask, tgt_mask,
            None,
            src_padding_mask, tgt_padding_mask,
            memory_key_padding_mask)
        out = self.out(out)
        out = self.AccentTokenizer(out)
        out = torch.logit(out, eps=1e-6)
        return out
