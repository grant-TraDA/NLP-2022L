from __future__ import unicode_literals, print_function, division
from tqdm import tqdm
import math
import time
import typing
import collections
import torch
import pathlib
import unicodedata
import re
import collections

from io import open
import unicodedata
import re

import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Transformer


MAX_LENGTH = 10
DATA_PATH = pathlib.Path('./data/forms/')


def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and \
        len(p[1].split()) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = s.lower()
    return s


SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3


class Tokenizer:
    def __init__(self, max_sentence_len):
        self.word_index = {"<SOS>": SOS_token, "<EOS>": EOS_token,
                           "<UNK>": UNK_token, "<PAD>": PAD_token}
        self.word_count = collections.Counter()
        self.n_words = max(self.word_index.values())+1
        self.sentence_len = max_sentence_len + 1   # EOS

    def train(self, list_of_words: typing.List[str]):
        self.word_count.update(list_of_words)
        for word in list_of_words:
            if word not in self.word_index:
                self.word_index[word] = self.n_words
                self.n_words += 1
        self.inverse_word_index = dict(
            [reversed(i) for i in self.word_index.items()])

    def tokenize(self, sentence: str):
        id_list = [self.word_index.get(word, UNK_token)
                   for word in sentence.split()] + [EOS_token]
        id_list += [PAD_token for _ in range(self.sentence_len - len(id_list))]
        return torch.tensor(id_list, dtype=torch.long, device=device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, d_model, nhead, dim_feedforward, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        # self.src_mask = None
        self.emb = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.trans = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers,
                                 num_decoder_layers=nlayers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.out = nn.Linear(d_model, ntoken)
        self.d_model = d_model

    def encode(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src = self.encode(src)
        tgt = self.encode(tgt)
        out = self.trans(src, tgt, src_mask, tgt_mask, None,
                         src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        out = self.out(out)
        out = F.log_softmax(out, dim=-1)
        return out


def test_on_line(model, temp=1, zero_line_text="this racks the joints this fires the veins"):
    zero_line = t_tokenizer.tokenize(zero_line_text).reshape(MAX_LENGTH+1, 1)
    # print(zero_line.shape)
    next_line = torch.tensor([SOS_token]*(MAX_LENGTH+1),
                             dtype=torch.long, device=device).reshape(MAX_LENGTH+1, 1)
    next_line_id = 1
    # print(next_line)

    for i in range(MAX_LENGTH-1):
        src_padding_mask = padding_mask(zero_line)
        tgt_padding_mask = padding_mask(next_line)

        output = model(zero_line, next_line, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        # topv, topi = output[next_line_id].data.topk(1)
        topi = torch.multinomial(
            output[next_line_id].squeeze().div(temp).exp().cpu(), 1)[0]
        res_id = topi.item()
        res = t_tokenizer.inverse_word_index[res_id]
        next_line[next_line_id] = res_id
        next_line_id += 1
    return " ".join([
        t_tokenizer.inverse_word_index[output_id.item()]
        for output_id in next_line
    ])


def padding_mask(x):
    x_padding_mask = (x == PAD_token).transpose(0, 1)
    return x_padding_mask


def prepare_tgt(tgt):
    tgt_input = torch.cat((torch.ones(
        (1, tgt.shape[1]), device=device)*SOS_token, tgt[:-1, :])).type(torch.long)
    tgt_output = torch.cat((tgt[1:, :], torch.ones(
        (1, tgt.shape[1]), device=device)*EOS_token)).type(torch.long)
    return tgt_input, tgt_output


def train():
    total_loss = 0
    for batch in tqdm(train_dataloader):
        src, tgt = batch[0], batch[1]
        src = src.transpose(0, 1).type(torch.long)
        tgt = tgt.transpose(0, 1).type(torch.long)
        tgt_input, tgt_output = prepare_tgt(tgt)
        src_padding_mask = padding_mask(src)
        tgt_padding_mask = padding_mask(tgt_input)

        output = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        output = output.reshape(-1, output.shape[-1])

        opt.zero_grad()
        loss = loss_fn(output, tgt_output.reshape(-1))
        loss.backward()

        # pred = pred.permute(1, 2, 0)

        opt.step()

        total_loss += loss.item()
    return total_loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)
    print(torch.__version__)
    dataset = []
    for dirname in DATA_PATH.iterdir():
        # if 'abc' not in dirname.name:
        #     continue
        for file_name in (dirname).iterdir():
            line0 = ''
            with open(file_name, encoding='utf-8') as f:
                for line in f:
                    line = normalizeString(line)
                    if line0 == '':
                        line0 = line
                        continue
                    dataset.append([line0, line])
                    line0 = ''
    dataset = filterPairs(dataset)
    print(len(dataset))

    t_tokenizer = Tokenizer(MAX_LENGTH)
    t_tokenizer.train(' '.join([
        " ".join(pair)
        for pair in dataset
    ]).split())

    torch_dataset = [
        (t_tokenizer.tokenize(pair[0]), t_tokenizer.tokenize(pair[1]))
        for pair in dataset
    ]
    train_size = int(0.8*len(torch_dataset))
    test_size = len(torch_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        torch_dataset, [train_size, test_size])
    data_loader_kwargs = {'num_workers': 1,
                          'pin_memory': True} if device == 'cuda' else {}
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        **data_loader_kwargs
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=20,
        shuffle=True,
        **data_loader_kwargs
    )

    d_model = 200  # word embeddings
    nhead = 2  # number of heads in the encoder/decoder of the transformer model
    dim_feedforward = 200  # hidden units per layer
    nlayers = 2  # num of layers
    dropout = 0.2
    model = TransformerModel(t_tokenizer.n_words, d_model,
                             nhead, dim_feedforward, nlayers, dropout).to(device)


    tgt_mask = (1-torch.ones((MAX_LENGTH+1, MAX_LENGTH+1),
                             device=device).tril()).type(torch.bool)
    print(tgt_mask)

    src_mask = torch.zeros((MAX_LENGTH+1, MAX_LENGTH+1),
                           device=device).type(torch.bool)
    print(src_mask)

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_token)

    best_loss = None
    epoches = 1024
    for epoch in range(1, epoches+1):
        epoch_start_time = time.time()
        train_loss = train()
        # val_loss = evaluate(test_dataset)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'
              .format(epoch, (time.time() - epoch_start_time), train_loss))
        print('-' * 89)
        print("Test of single line")
        print(f'hot answer: {test_on_line(model, temp=0.8)}')
        print(f'cold answer: {test_on_line(model, temp=0.1)}')
        # Save the model if the validation loss is the best we've seen so far.
        if not best_loss or train_loss < best_loss:
            print("Best Score, YEY!")
            best_loss = train_loss

    with open('basic_saved_model', 'wb') as f:
        torch.save(model, f)
