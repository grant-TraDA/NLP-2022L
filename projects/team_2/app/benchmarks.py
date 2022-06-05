import os
import random
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer

from src.preprocessing import load_data
from src.summarization import summarize_multiple
import pandas as pd
from typing import List


def run_and_save(n_iter: int=250, length: int=4, capacity: float=.1, model: SentenceTransformer=SentenceTransformer('all-MiniLM-L6-v2'),
                 summary_path: str="summary", targets_path: str="targets", subset_size: int=None) -> List[str]:
    """Run summaries on a set of aricles and saves results to adequate files.

    Args:
        n_iter (int, optional): maximum number of iterations of the PSO algorithm. Defaults to 250.
        length (int, optional): expected length of a summary. Defaults to 4.
        capacity (float, optional): parameter defining how much pso will use exceeding size penality, bigger capacity allows more exceeding summary length (0 - only penality, 1 - only similarity). Defaults to .1.
        model (SentenceTransformer, optional): embedding model from sentence_transformers package. Defaults to SentenceTransformer('all-MiniLM-L6-v2').
        summary_path (str, optional): path to a file where summaries should be saved. Defaults to "summary".
        targets_path (str, optional): path to a file where targets should be saved. Defaults to "targets".
        subset_size (int, optional): size of a subset of the whole data. Defaults to None.

    Returns:
        List[str]: list of summaries for following articles
    """
    _, _, test = load_data("src/data")

    # choosing a subset from training data (not necessarily)
    if subset_size is None:
        subset = range(len(test))
    else:
        subset = random.sample(range(len(test)), subset_size)

    # calculacting summaries for all articles
    summaries: List[str] = summarize_multiple(test.article[subset], model=model, n_iter=n_iter, length=length, capacity=capacity)
    
    # below process is needed for the purpose of calculating rouge metrics with which we compare our model to SOA examples

    # getting target summaries from the dataset
    target_summaries: pd.Series = test.highlights[subset]

    # saving our summaries
    for i, s in zip(subset, summaries):
        path = Path(summary_path)
        os.makedirs(path, exist_ok=True)
        with open(path / (str(i).rjust(5, "0") + ".txt"), "w", encoding="utf-8") as f:
            f.write(s)

    # saving target summaries
    for i, s in zip(subset, target_summaries):
        path = Path(targets_path)
        os.makedirs(path, exist_ok=True)
        with open(path / (str(i).rjust(5, "0") + ".txt"), "w", encoding="utf-8") as f:
            f.write(s)

    return summaries


if __name__ == '__main__':
    # grid search along parameters
    capacities = np.arange(0.1, 1, 0.2)
    models = ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-distilroberta-v1', 'all-mpnet-base-v2']
    for model_name in models:
        model = SentenceTransformer(model_name)
        for capacity in capacities:
            run_and_save(capacity=capacity,
                         model=model,
                         subset_size=1,
                         summary_path=f"summaries/summary_{int(capacity*100)}_{model_name}",
                         targets_path="src/targets")
            break
        break
