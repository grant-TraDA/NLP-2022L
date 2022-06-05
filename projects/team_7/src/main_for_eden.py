from __future__ import unicode_literals, division
from tqdm import tqdm
import time
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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



def test_gpt2_on_line(gpt2, t_tokenizer, temp=1, zero_line_text="this definitely racks the joints this fires the veins"):
    next_line = ""
    next_line_id = 0
    for _ in range(config.MAX_LENGTH-1):
        gtp2_in = torch.tensor(
            t_tokenizer.encode(zero_line_text+next_line), device=config.DEVICE).unsqueeze(0)
        gpt2_output = gpt2(gtp2_in).logits
        gpt2_output = F.softmax(gpt2_output[0, -1].squeeze().div(temp), dim=-1).cpu()
        best_output, best_output_indices = torch.topk(gpt2_output, 20)
        topi = torch.multinomial(best_output, 1)[0]
        res_id = best_output_indices[topi.item()]
        next_line += t_tokenizer.decode(res_id)
        next_line_id += 1
    return next_line


def test_model_on_line(model, t_tokenizer, a_tokenizer, temp=1, zero_line_text="this definitely racks the joints this fires the veins"):
    next_line = ""
    src_mask, tgt_mask = utils.create_basic_masks()
    next_line_id = 0
    accent_patterns = ''
    for _ in range(config.MAX_LENGTH-1):
        src, tgt, _, _ = utils.create_input_data(
            zero_line_text, next_line, t_tokenizer, a_tokenizer)
        if tgt.shape[0] > config.MAX_LENGTH:
            return next_line
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)
        model_output = model(
            src, tgt_input,
            src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask,
            src_padding_mask)
        model_output = model_output[next_line_id].squeeze()
        output = F.softmax(model_output.div(temp), dim=-1).cpu()
        best_output, best_output_indices = torch.topk(output, 20)

        topi = torch.multinomial(best_output, 1)[0]
        res_id = best_output_indices[topi.item()]
        detokenized = a_tokenizer.detokenize(res_id)
        next_line = next_line + ' ' + a_tokenizer.detokenize(res_id)
        next_line_id += 1
        accent_patterns = accent_patterns + ' ' + a_tokenizer.get_accent_pattern(res_id.item()) 

    logger.debug(f'First line: {zero_line_text}')
    logger.debug(f'src: {src.tolist()}')
    logger.debug(f'Second line: {next_line}')
    logger.debug(f'tgt: {tgt.tolist()}')

    return accent_patterns


def test_model_and_gpt2_on_line(model, gpt2, t_tokenizer, a_tokenizer, temp=1, zero_line_text="this definitely racks the joints this fires the veins"):
    next_line = ""
    src_mask, tgt_mask = utils.create_basic_masks()
    next_line_id = 0
    for _ in range(config.MAX_LENGTH-1):
        src, tgt, gtp2_in, _ = utils.create_input_data(
            zero_line_text, next_line, t_tokenizer, a_tokenizer)
        gtp2_in = torch.tensor(
            t_tokenizer.encode(gtp2_in), device=config.DEVICE).unsqueeze(0)
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)
        model_output = model.get_matrix(
            src, tgt_input,
            src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask,
            src_padding_mask)
        model_output = model_output[next_line_id].squeeze()

        gpt2_output = gpt2(gtp2_in).logits[0, -1].squeeze()
        output = model_output + gpt2_output
        output = F.softmax(output.div(temp), dim=-1).cpu()
        best_output, best_output_indices = torch.topk(output, 20)

        # for debug
        gpt2_output = F.softmax(gpt2_output.div(temp), dim=-1).cpu()
        best_gpt2_output, best_gpt2_output_indices = torch.topk(gpt2_output, 20)
        best_words_gpt2 = [t_tokenizer.decode(i) for i in best_gpt2_output_indices]
        best_words = [t_tokenizer.decode(i) for i in best_output_indices]
        # logger.debug(f"Best words before accents: {best_words_gpt2}")
        # logger.debug(f"Best words after accents: {best_words}")

        topi = torch.multinomial(best_output, 1)[0]
        res_id = best_output_indices[topi.item()]
        next_line += t_tokenizer.decode(res_id)
        next_line_id += 1
    return next_line


def train_model(model, train_dataloader):
    opt = torch.optim.SGD(model.parameters(), lr=config.MODEL_LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_token)

    src_mask, tgt_mask = utils.create_basic_masks()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training model"):
        src, tgt = batch[0], batch[1]
        src = src.squeeze(-1).transpose(0, 1)
        tgt = tgt.squeeze(-1).transpose(0, 1)
        # tgt = tgt.transpose(0, 1).type(torch.long)
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)

        output = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask)

        output = output.reshape(-1, output.shape[-1])

        opt.zero_grad()
        loss = loss_fn(output, tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

        opt.step()
        total_loss += loss.item()
    return total_loss



def train_gpt2(gpt2, t_tokenizer, train_dataloader):
    gpt2.train()
    opt = torch.optim.AdamW(gpt2.parameters(), lr=config.GPT2_LR)
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training model"):
        gtp2_ins = batch[2]
        opt.zero_grad()

        for b in range(len(gtp2_ins)):
            gtp2_in = torch.tensor(
                t_tokenizer.encode(gtp2_ins[b]), device=config.DEVICE).unsqueeze(0)
            gpt2_loss = gpt2(gtp2_in).loss
            gpt2_loss.backward()
            total_loss += gpt2_loss.detach().data
        torch.nn.utils.clip_grad_norm_(gpt2.parameters(), 0.25)
        opt.step()
    return total_loss


def train_model_and_gpt2(model, gpt2, t_tokenizer, train_dataloader):
    gpt2.train()
    opt = torch.optim.SGD(gpt2.parameters(), lr=config.GPT2_LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_token)
    src_mask, tgt_mask = utils.create_basic_masks()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training model"):
        src, tgt, gtp2_ins, nums_tokens_first_line = batch[0], batch[1], batch[2], batch[3]

        src = src.transpose(0, 1).squeeze(-1).type(torch.long)
        tgt = tgt.transpose(0, 1).squeeze(-1).type(torch.long)
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)
        output = model.get_matrix(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask).transpose(0,1) # should be 3d
        # output = output.reshape(-1, output.shape[-1])

        loss = None
        opt.zero_grad()
        b = 0
        for gtp2_lines, num_tokens_first_line in zip(gtp2_ins, nums_tokens_first_line):
            gtp2_in = torch.tensor(
                t_tokenizer.encode(gtp2_lines), device=config.DEVICE).unsqueeze(0)
            gpt2_output = gpt2(gtp2_in).logits
            gpt2_output = gpt2_output[:, int(num_tokens_first_line):, :]
            num_tokens_second_line = gpt2_output.shape[1]
            gpt2_output = F.softmax(gpt2_output, dim=-1).squeeze()
            final_result = (
                torch.mul(output[b, :num_tokens_second_line, :], gpt2_output)
                ).transpose(0,1).unsqueeze(0)
            b += 1
            ground_truth = gtp2_in[:, int(num_tokens_first_line):]
            if not loss:
                loss = loss_fn(final_result, ground_truth)
            else:
                loss = loss + loss_fn(final_result, ground_truth)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt2.parameters(), 0.25)
        opt.step()
        total_loss += loss.item()
    return total_loss


def test_model(model, test_dataloader):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_token)
    src_mask, tgt_mask = utils.create_basic_masks()
    total_loss = 0
    for batch in tqdm(test_dataloader, desc="Testing model"):

        src, tgt = batch[0], batch[1]
        src = src.transpose(0, 1).squeeze(-1).type(torch.long)
        tgt = tgt.transpose(0, 1).squeeze(-1).type(torch.long)
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)
        output = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask).transpose(0,1)

        output = output.reshape(-1, output.shape[-1])
        loss = loss_fn(output, tgt_output.reshape(-1))
        total_loss += loss.item()
    return total_loss


def test_model_and_gpt2(model, gpt2, t_tokenizer, test_dataloader):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_token)
    src_mask, tgt_mask = utils.create_basic_masks()
    total_loss = 0
    for batch in tqdm(test_dataloader, desc="Testing model and gpt2"):
        src, tgt, gtp2_ins, nums_tokens_first_line = batch[0], batch[1], batch[2], batch[3]

        src = src.transpose(0, 1).squeeze(-1).type(torch.long)
        tgt = tgt.transpose(0, 1).squeeze(-1).type(torch.long)
        tgt_input, tgt_output = utils.prepare_tgt(tgt)
        src_padding_mask = utils.padding_mask(src)
        tgt_padding_mask = utils.padding_mask(tgt_input)
        output = model.get_matrix(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask).transpose(0,1) # should be 3d
        # output = output.reshape(-1, output.shape[-1])

        loss = None
        b = 0
        for gtp2_lines, num_tokens_first_line in zip(gtp2_ins, nums_tokens_first_line):
            gtp2_in = torch.tensor(
                t_tokenizer.encode(gtp2_lines), device=config.DEVICE).unsqueeze(0)
            gpt2_output = gpt2(gtp2_in).logits
            gpt2_output = gpt2_output[:, int(num_tokens_first_line):, :]
            num_tokens_second_line = gpt2_output.shape[1]
            gpt2_output = F.softmax(gpt2_output, dim=-1).squeeze()
            final_result = (
                torch.mul(output[b, :num_tokens_second_line, :], gpt2_output)
                ).transpose(0,1).unsqueeze(0)
            b += 1
            ground_truth = gtp2_in[:, int(num_tokens_first_line):]
            if not loss:
                loss = loss_fn(final_result, ground_truth)
            else:
                loss = loss + loss_fn(final_result, ground_truth)

        total_loss += loss.item()
    return total_loss


def load_models():

    logger.info("Loading tokenizers")
    t_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    t_tokenizer.num_words = max(t_tokenizer.decoder)+1

    a_tokenizer = accent_tokenizer.AccentTokenizer.create(
        t_tokenizer, config.ROOT_PATH / "tmp.pickle")

    logger.debug('Tokenizer accent patterns')
    logger.debug(f'{a_tokenizer.accent_patterns}')

    d_model = 200  # word embeddings
    nhead = 2  # number of heads in the encoder/decoder of the transformer model
    dim_feedforward = 200  # hidden units per layer
    nlayers = 2  # num of layers
    dropout = 0.2
    logger.info("Creating model")

    gpt2_path = config.ROOT_PATH / "saved_gpt2"
    if gpt2_path.exists():
        with open(gpt2_path, 'rb') as f:
            gpt2 = transformer_model.Gtp2Model().to(config.DEVICE)
            gpt2.load_state_dict(torch.load(f))
    else:
        gpt2 = transformer_model.Gtp2Model().to(config.DEVICE)

    model_path = config.ROOT_PATH / "saved_model"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model = transformer_model.PoemModel(
                a_tokenizer.num_words,
                d_model,
                nhead,
                dim_feedforward,
                nlayers,
                a_tokenizer.matrix,
                dropout
            ).to(config.DEVICE)
            model.load_state_dict(torch.load(f))
    else:
        model = transformer_model.PoemModel(
            a_tokenizer.num_words,
            d_model,
            nhead,
            dim_feedforward,
            nlayers,
            a_tokenizer.matrix,
            dropout
        ).to(config.DEVICE)
    return model, gpt2, t_tokenizer, a_tokenizer


def load_data(t_tokenizer, a_tokenizer):
    dataset = pd.read_csv(config.DATA_PATH)

    dataset = dataset.values.tolist()
    logger.info(len(dataset))


    logger.info("\nTokenizing, preparing dataset")

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

    logger.info("Filtering out too long records")
    logger.info(f"torch_dataset before filtering: {len(torch_dataset)}")
    torch_dataset = [
        x for x in torch_dataset
        if x[0].shape[0] == config.MAX_LENGTH
        and x[1].shape[0] == config.MAX_LENGTH
        ]
    logger.info(f"torch_dataset after filtering: {len(torch_dataset)}")

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

    return train_dataloader, test_dataloader

def main_learn_model(model, t_tokenizer, a_tokenizer, train_dataloader):
    logger.info("Learning accent prediction")
    best_loss = None
    epoches = 1
    for epoch in range(1, epoches+1):
        logger.info("Test of single line")
        logger.info(f'hot model answer: {test_model_on_line(model, t_tokenizer, a_tokenizer, temp=0.8)}')
        logger.info(f'cold model answer: {test_model_on_line(model, t_tokenizer, a_tokenizer, temp=0.1)}')
        epoch_start_time = time.time()
        train_loss = train_model(model, train_dataloader)
        logger.info(f'| end of epoch {epoch} | time: {time.time() - epoch_start_time}s | train loss {train_loss}')
        if not best_loss or train_loss < best_loss:
            logger.info("Best Score, YEY!")
            best_loss = train_loss

    with open(config.ROOT_PATH / 'saved_model', 'wb') as f:
        torch.save(model.state_dict(), f)
    test_loss = test_model(model, test_dataloader)
    logger.info(f"test loss: {test_loss}")


def main_learn_gpt2(gpt2, t_tokenizer, train_dataloader):
    logger.info("Fine-tuning gpt2")
    best_loss = None
    epoches = 20
    for epoch in range(1, epoches+1):
        logger.info("Test of single line")
        logger.info(f'hot gpt2 answer: {test_gpt2_on_line(gpt2, t_tokenizer, temp=0.8)}')
        logger.info(f'cold gpt2 answer: {test_gpt2_on_line(gpt2, t_tokenizer, temp=0.1)}')
        epoch_start_time = time.time()
        train_loss = train_gpt2(gpt2, t_tokenizer, train_dataloader)
        logger.info(f'| end of epoch {epoch} | time: {time.time() - epoch_start_time}s | train loss {train_loss}')
        if not best_loss or train_loss < best_loss:
            logger.info("Best Score, YEY!")
            best_loss = train_loss
        with open(config.ROOT_PATH / 'saved_gpt2', 'wb') as f:
            torch.save(gpt2.state_dict(), f)
    test_loss = test_model_and_gpt2(model, gpt2, t_tokenizer, test_dataloader)
    logger.info(f"test loss: {test_loss}")


def compare_test_results(model, gpt2, t_tokenizer, a_tokenizer, test_dataloader):
    for batch in tqdm(test_dataloader, desc="Comparing with/without model"):
        lines = batch[2]
        for line in lines:
            line0 = line[:len(line)//2]
            line1 = line[len(line)//2:]
            result_both = test_model_and_gpt2_on_line(
                model, gpt2, t_tokenizer, a_tokenizer, 0.8, zero_line_text=line0)
            logger.debug(f"First line: {line0}")
            logger.debug("High temperature:")
            logger.debug(f"\tWith model: {result_both}")
            result_gpt2 = test_gpt2_on_line(gpt2, t_tokenizer, 0.8, zero_line_text=line0)
            logger.debug(f"\tNo model:   {result_gpt2}")
            logger.debug("Low temperature:")
            result_both = test_model_and_gpt2_on_line(
                model, gpt2, t_tokenizer, a_tokenizer, 0.1, zero_line_text=line0)
            logger.debug(f"\tWith model: {result_both}")
            result_gpt2 = test_gpt2_on_line(gpt2, t_tokenizer, 0.1, zero_line_text=line0)
            logger.debug(f"\tNo model:   {result_gpt2}")


if __name__ == '__main__':
    logger.info(config.DEVICE)
    logger.info(torch.__version__)
    model, gpt2, t_tokenizer, a_tokenizer = load_models()
    train_dataloader, test_dataloader = load_data(t_tokenizer, a_tokenizer)

    # main_learn_model(model, t_tokenizer, a_tokenizer, train_dataloader)
    # main_learn_gpt2(gpt2, t_tokenizer, train_dataloader)
    compare_test_results(model, gpt2, t_tokenizer, a_tokenizer, test_dataloader)
