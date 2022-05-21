import nlpaug
import nlpaug.flow
import nlpaug.augmenter.word as naw
from nltk.tokenize import RegexpTokenizer
import numpy as np

class WordMask(nlpaug.Augmenter):
    def __init__(self,
                 tokenizer,
                 aug_p):
        self.mask = 'ż'
        self.tokenizer = tokenizer
        self.aug_p = aug_p

    def augment(self, x, n=0, num_thread=0):
        t = self.tokenizer(x)
        n = np.random.randint(max(self.aug_p * len(t), 1))
        c = np.random.choice(range(len(t)), n)
        for c in c:
            t[c] = self.mask
        return " ".join(t)

    def is_duplicate(self, a, b):
        return False


def augmenter(
        aug_p_swap=0.001,
        aug_p_mask=0.3,
        aug_min=0,
        aug_max=None):
    aug_swap = naw.RandomWordAug(
        action="swap",
        aug_p=aug_p_swap,
        aug_min=aug_min,
        aug_max=aug_max,
        tokenizer=RegexpTokenizer(r'\S+').tokenize
    )

    aug_wordmask = WordMask(
        aug_p=aug_p_mask,
        tokenizer=RegexpTokenizer(r'\S+').tokenize
    )
    
    aug = nlpaug.flow.Sequential([
        aug_swap,
        aug_wordmask,
    ])
    return aug
