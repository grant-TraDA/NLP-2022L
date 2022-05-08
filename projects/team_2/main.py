import os
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer

from preprocessing import load_data
from summarization import summarize_multiple


def run_and_save(n_iter=250, length=4, capacity=.1, model=SentenceTransformer('all-MiniLM-L6-v2'),
                 summary_path="summary", targets_path="targets", subset_size=None):
    _, _, test = load_data("data")

    if subset_size is None:
        subset = range(len(test))
    else:
        subset = random.sample(range(len(test)), subset_size)

    summaries = summarize_multiple(test.article[subset], model=model, n_iter=n_iter, length=length, capacity=capacity)
    target_summaries = test.highlights[subset]

    for i, s in zip(subset, summaries):
        path = Path(summary_path)
        os.makedirs(path, exist_ok=True)
        with open(path / (str(i).rjust(5, "0") + ".txt"), "w", encoding="utf-8") as f:
            f.write(s)

    for i, s in zip(subset, target_summaries):
        path = Path(targets_path)
        os.makedirs(path, exist_ok=True)
        with open(path / (str(i).rjust(5, "0") + ".txt"), "w", encoding="utf-8") as f:
            f.write(s)

    return summaries


if __name__ == '__main__':
    run_and_save()
