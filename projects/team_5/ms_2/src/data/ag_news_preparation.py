
from data.autoencoding_dataset import AutoencodingDataset
from augmentation.augmentation import augmenter
from torchtext.datasets import AG_NEWS
import re

def AG_NEWS_preparation(split="train", max_len=16, aug_params=False):
    """AG_NEWS_preparation
    
    Preparation of the AG-News dataset.
    
    Parameters:
        split="train": "test" or "train";
        max_len=16: maximum length of a sequence;
        aug_params=False: params for augmentation. If False, no augmentation is applied during training.
    
    Returns:
        Dataset of sentences.
    
    """
    sentences = [
        re.sub(r'lt|gt|href|fullquote|aspx', ' ', d[1].replace('-', " "))
        for d in list(AG_NEWS(split=split))
    ]
    aug = augmenter(**aug_params).augment if aug_params != False else False
    dp = AutoencodingDataset(sentences, aug, max_len=max_len)
    dp.preproces()
    return dp
