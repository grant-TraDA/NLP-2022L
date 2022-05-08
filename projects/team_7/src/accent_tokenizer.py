from tqdm import tqdm
import pathlib
from pysle import isletool
import pickle

import torch
import torch.nn as nn
import numpy as np

import config
import utils


class AccentTokenizer:
    @classmethod
    def create(cls, t_tokenizer, filename: pathlib.Path):
        '''
        Tries to read AccentTokenizer from file if filename exists.
        If not creates AccentTokenizer and saves to file
        '''
        if filename.exists():
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            return model

        a_tokenizer = cls(t_tokenizer, filename)

        with open(filename, 'wb') as f:
            model = pickle.dump(a_tokenizer, f, pickle.HIGHEST_PROTOCOL)

        return a_tokenizer

    def __init__(self, t_tokenizer):
        '''
        Creates dictionary self.word_to_accent_id subword_id -> accent_type_id 
        where subword_id is id of subword based on GPT2Tokenizer
        and accent_type_id is id of accent type.
        Creates matrix self.matrix which can be used to change accent_type_id of entity into correct subword_id
        '''
        isle = isletool.Isle()
        word_to_accent_pattern = {}
        word_to_gtp2_ids = {}
        for tokens in tqdm(t_tokenizer.__dict__['bpe_ranks'].keys(), desc="Initialising AccentTokenizer part 1"):
            word = utils.join_gtp2_tokens_to_word(tokens)
            word_to_gtp2_ids[word] = [t_tokenizer.encoder[t] for t in tokens]
            try:
                entry = isle.lookup(word)[0]
            except:
                word_to_accent_pattern[word] = ""
                continue
            result = []
            subwords = [subword for subword in entry.syllabificationList]
            if len(subwords) != 1:
                continue
            syllabes = subwords[0]
            for syllabe in syllabes.toList():
                stressed = False
                for character in syllabe:
                    if character[0] == 'ˈ':
                        stressed = True
                result.append(stressed)
            output = "".join(["A" if s else "a" for s in result])
            word_to_accent_pattern[word] = output

        word_to_accent_pattern["<sos>"] = ""
        word_to_accent_pattern["<eos>"] = ""
        word_to_accent_pattern["<unk>"] = ""
        word_to_accent_pattern["<pad>"] = ""
        unique_accent_patterns = list(set(word_to_accent_pattern.values()))
        num_patterns = len(unique_accent_patterns)
        accent_pattern_to_id = {k: v+3 for k, v in zip(
            unique_accent_patterns, range(num_patterns))}
        self.num_words = num_patterns+3
        word_to_accent_id = {word: accent_pattern_to_id[pattern]
                             for word, pattern in word_to_accent_pattern.items()}

        self.no_accent_id = accent_pattern_to_id[""]

        matrix = np.zeros((t_tokenizer.num_words, self.num_words))
        for gtp2_id in tqdm(t_tokenizer.decoder.keys(), desc="Initialising AccentTokenizer part 2"):
            for word, ids in word_to_gtp2_ids.items():
                if gtp2_id in ids:
                    accent = word_to_accent_id.get(
                        word, self.no_accent_id)
                    matrix[gtp2_id, accent] = 1

        self.word_to_accent_id = word_to_accent_id
        self.matrix = matrix

    def tokenize(self, sentence: str, t_tokenizer):
        '''
        Generates padded list of accent types in sentence
        '''
        tokenized = t_tokenizer.encode(sentence)
        id_list = []
        part_of_last_word = False
        for i, token_id1 in enumerate(tokenized[:-1]):
            if part_of_last_word:
                id_list.append(id_list[-1])
                part_of_last_word = False
                continue
            token_id2 = tokenized[i+1]
            token1 = t_tokenizer.decode(token_id1)
            token2 = t_tokenizer.decode(token_id2)
            if (token1, token2) in t_tokenizer.__dict__['bpe_ranks']:
                word = utils.join_gtp2_tokens_to_word([token1, token2])
                part_of_last_word = True
            else:
                word = token1
            accent_id = self.word_to_accent_id.get(word, self.no_accent_id)
            id_list.append(accent_id)
        if sentence:
            if part_of_last_word:
                id_list.append(id_list[-1])
            else:
                accent_id = self.word_to_accent_id.get(
                    t_tokenizer.decode(tokenized[-1]), self.no_accent_id)
                id_list.append(accent_id)

        id_list += [config.PAD_token for _ in range(
            config.MAX_LENGTH - len(id_list))]
        return torch.tensor(id_list, dtype=torch.long, device=config.DEVICE)


class AccentToGtp2Tokenizer(nn.Module):
    def __init__(self, matrix):
        '''
        Uses matrix from AccentTokenizer to transform tensor of accent type ids into tensor of subword ids.
        '''
        super(AccentToGtp2Tokenizer, self).__init__()
        accent_to_word_transform = torch.from_numpy(matrix).T.type(torch.float)
        self.register_buffer('accent_to_word_transform',
                             accent_to_word_transform)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, accent_types]
        Outputs:
            Tensot, shape [seq_len, batch_size, subwords]
        """
        return torch.matmul(x, self.accent_to_word_transform)
