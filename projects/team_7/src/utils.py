import torch

import config


def padding_mask(x):
    '''
    Creates padding mask for matrix x
    '''
    x_padding_mask = (x == config.PAD_token).transpose(0, 1)
    return x_padding_mask


def prepare_tgt(tgt):
    '''
    Creates input and output vectors for transformer model
    Shifts input vector one position to the right and inserts Start of Sentence token at the beginning
    '''
    tgt_input = torch.cat((torch.ones(
        (1, tgt.shape[1]), device=config.DEVICE)*config.SOS_token, tgt[:-1, :])).type(torch.long)
    # tgt_output = torch.cat((tgt[1:, :], torch.ones(
    #     (1, tgt.shape[1]), device=device)*EOS_token)).type(torch.long)
    tgt_output = tgt.type(torch.long)
    return tgt_input, tgt_output


def create_input_data(line0, line1, t_tokenizer, a_tokenizer):
    '''
    Performs tokenizing and padding of input data 
    '''
    num_tokens_first_line = len(t_tokenizer.encode(line0))
    two_lines = line0 + " " + line1
    t_tokenized = t_tokenizer.encode(two_lines)
    t_tokenized_line0 = t_tokenized[:num_tokens_first_line]
    t_tokenized_line1 = t_tokenized[num_tokens_first_line:]
    gtp2_in = two_lines
    model_in_line0 = a_tokenizer.tokenize(t_tokenized_line0, t_tokenizer)
    model_in_line1 = a_tokenizer.tokenize(t_tokenized_line1, t_tokenizer)
    return (
        model_in_line0.unsqueeze(-1).type(torch.long),
        model_in_line1.unsqueeze(-1).type(torch.long),
        gtp2_in,
        str(num_tokens_first_line)
    )


def create_basic_masks():
    '''
    Creates zeros mask for src and lower triangular matrix for tgt
    '''
    src_mask = torch.zeros((config.MAX_LENGTH, config.MAX_LENGTH),
                           device=config.DEVICE).type(torch.bool)
    tgt_mask = (1-torch.ones((config.MAX_LENGTH, config.MAX_LENGTH),
                             device=config.DEVICE).tril()).type(torch.bool)
    return (src_mask, tgt_mask)


def join_gtp2_tokens_to_word(tokens):
    '''
    Merges subwords into single word
    '''
    return "".join(tokens).replace('Ġ', '').replace(" ", '').lower()
