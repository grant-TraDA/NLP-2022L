import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from preprocessing import load_data
from summarization import summarize_multiple


def run_and_save(n_iter=250, length=4, capacity=.1, model=SentenceTransformer('all-MiniLM-L6-v2'),
                 summary_path="summary", targets_path="targets"):
    train, valid, test = load_data("data")

    summaries = summarize_multiple(test.article, model=model, n_iter=n_iter, length=length, capacity=capacity)
    target_summaries = test.highlights

    for i, s in enumerate(summaries):
        path = Path(summary_path)
        os.makedirs(path, exist_ok=True)
        with open(path / (str(i).rjust(5, "0") + ".txt"), "w", encoding="utf-8") as f:
            f.write(s)

    for i, s in enumerate(target_summaries):
        path = Path(targets_path)
        os.makedirs(path, exist_ok=True)
        with open(path / (str(i).rjust(5, "0") + ".txt"), "w", encoding="utf-8") as f:
            f.write(s)

    return summaries


if __name__ == '__main__':
    run_and_save()
