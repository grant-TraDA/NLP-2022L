
from data.autoencoding_dataset import AutoencodingDataset
from augmentation.augmentation import augmenter
from torchtext.datasets import IMDB
import re

def IMDB_preparation(split="train", max_len=16, aug_params=False):
    """IMDB_preparation
    
    Preparation of the IMDB dataset.
    
    Parameters:
        split="train": "test" or "train";
        max_len=16: maximum length of a sequence;
        aug_params=False: params for augmentation. If False, no augmentation is applied during training.
    
    Returns:
        Dataset of sentences.
    
    """
    sentences = [
        re.sub(r'<\s*br\s*/>', ' ', d[1].replace('-', " "))
        for d in list(IMDB(split=split))
    ]
    aug = augmenter(**aug_params).augment if aug_params != False else False
    dp = AutoencodingDataset(sentences, aug, max_len=max_len)
    dp.preproces()
    return dp
