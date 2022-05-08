
from data.autoencoding_dataset import AutoencodingDataset
from augmentation.augmentation import augmenter
from torchtext.datasets import IMDB
import re

def IMDB_preparation(split="train", max_len=16, aug_params=False):
    sentences = [
        re.sub(r'<\s*br\s*/>', ' ', d[1].replace('-', " "))
        for d in list(IMDB(split=split))
    ]
    aug = augmenter(**aug_params).augment if aug_params != False else False
    dp = AutoencodingDataset(sentences[:100], aug, max_len=max_len)
    dp.preproces()
    return dp
