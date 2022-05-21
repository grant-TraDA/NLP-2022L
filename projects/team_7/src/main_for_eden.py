from __future__ import unicode_literals, division
from tqdm import tqdm
import time
import pandas as pd
from transformers import GPT2Tokenizer
import pickle
import sys

import torch
import torch.utils.data
import torch.nn.functional as F

import transformer_model
import config
import utils
import accent_tokenizer

import logging

logger = logging.getLogger("Poem-Generation")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(
    config.LOG_PATH / 'main.log',
    encoding='utf-8'
)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def test_on_line(model, t_tokenizer, a_tokenizer, temp=1, zero_line_text="this racks the joints this fires the veins"):
    '''
    Check output of model based on one line of text
    '''
    src_mask, tgt_mask = utils.create_basic_masks()
    next_line = ""
    next_line_id = 0
    for _ in range(config.MAX_LENGTH-1):
        src, tgt, gtp2_in = utils.create_input_data(
            zero_line_text, next_line, t_tokenizer, a_tokenizer)
        src = src.unsqueeze(-1).type(torch.long)
        tgt = tgt.unsqueeze(-1).type(torch.long)
        gtp2_in = gtp2_in.unsqueeze(-1).type(torch.long)
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)

        output = model(
            src, tgt_input,
            src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask,
            src_padding_mask,
            gtp2_in
        )
        topi = torch.multinomial(
            F.softmax(
                output.transpose(0, 1)[next_line_id].squeeze().div(temp),
                dim=-1
            ).cpu(), 1
        )[0]

        res_id = topi.item()
        gtp2_in[next_line_id] = res_id
        next_line_id += 1
        next_line = t_tokenizer.decode(gtp2_in[config.MAX_LENGTH:config.MAX_LENGTH+next_line_id].flatten().tolist())
    return next_line


def train(model, train_dataloader):
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_token)

    src_mask, tgt_mask = utils.create_basic_masks()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training model"):

        src, tgt, gpt2_input = batch[0], batch[1], batch[2]
        src = src.transpose(0, 1).type(torch.long)
        tgt = tgt.transpose(0, 1).type(torch.long)
        gpt2_input = gpt2_input.transpose(0, 1).type(torch.long)
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)

        output = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask, gpt2_input)

        output = output.reshape(-1, output.shape[-1])

        opt.zero_grad()
        loss = loss_fn(output, tgt_output.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        # pred = pred.permute(1, 2, 0)

        opt.step()

        total_loss += loss.item()
    return total_loss


def test(model, test_dataloader):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_token)
    src_mask, tgt_mask = utils.create_basic_masks()
    total_loss = 0
    for batch in tqdm(test_dataloader, desc="Training model"):

        src, tgt, gpt2_input = batch[0], batch[1], batch[2]
        src = src.transpose(0, 1).type(torch.long)
        tgt = tgt.transpose(0, 1).type(torch.long)
        gpt2_input = gpt2_input.transpose(0, 1).type(torch.long)
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)

        output = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask, gpt2_input)

        output = output.reshape(-1, output.shape[-1])
        loss = loss_fn(output, tgt_output.reshape(-1))
        total_loss += loss.item()
    return total_loss


def main():
    dataset = pd.read_csv(config.DATA_PATH)

    dataset = dataset.values.tolist()
    logger.info(len(dataset))

    logger.info("Loading tokenizers")

    t_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    t_tokenizer.num_words = max(t_tokenizer.decoder)+1

    a_tokenizer = accent_tokenizer.AccentTokenizer.create(
        t_tokenizer, config.ROOT_PATH / "tmp.pickle")

    logger.info("Tokenizing, preparing dataset")

    dataset_path = config.ROOT_PATH / "tmp2.pickle"
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            torch_dataset = pickle.load(f)
    else:
        torch_dataset = [
            utils.create_input_data(pair[0], pair[1], t_tokenizer, a_tokenizer)
            for pair in tqdm(dataset, desc="Creating dataset")
        ]
        with open(dataset_path, 'wb') as f:
            model = pickle.dump(torch_dataset, f, pickle.HIGHEST_PROTOCOL)

    logger.info("Splitting dataset")

    train_size = int(0.8*len(torch_dataset))
    test_size = len(torch_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        torch_dataset, [train_size, test_size])
    data_loader_kwargs = {'num_workers': 1,
                          'pin_memory': True} if config.DEVICE == 'cuda' else {}
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=30,
        shuffle=True,
        **data_loader_kwargs
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=30,
        shuffle=True,
        **data_loader_kwargs
    )

    d_model = 200  # word embeddings
    nhead = 2  # number of heads in the encoder/decoder of the transformer model
    dim_feedforward = 200  # hidden units per layer
    nlayers = 2  # num of layers
    dropout = 0.2
    logger.info("Creating model")
    model = transformer_model.PoemModel(
        a_tokenizer.num_words,
        d_model,
        nhead,
        dim_feedforward,
        nlayers,
        a_tokenizer.matrix,
        dropout
    ).to(config.DEVICE)

    best_loss = None
    epoches = 1024
    for epoch in range(1, epoches+1):
        epoch_start_time = time.time()
        train_loss = train(model, train_dataloader)
        # val_loss = evaluate(test_dataset)
        logger.info(f'| end of epoch {epoch} | time: {time.time() - epoch_start_time}s | valid loss {train_loss}')
        logger.info("Test of single line")
        logger.info(f'hot answer: {test_on_line(model, t_tokenizer, a_tokenizer, temp=0.8)}')
        logger.info(f'cold answer: {test_on_line(model, t_tokenizer, a_tokenizer ,temp=0.1)}')

        if not best_loss or train_loss < best_loss:
            # with open(args.save, 'wb') as f:
            #     torch.save(model, f)
            logger.info("Best Score, YEY!")
            best_loss = train_loss

    with open(config.ROOT_PATH / 'saved_model', 'wb') as f:
        torch.save(model, f)

    test_loss = test(model, test_dataloader)
    logger.info(f"test loss: {test_loss}")


if __name__ == '__main__':
    logger.info(config.DEVICE)
    logger.info(torch.__version__)
    main()
