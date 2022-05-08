from preprocessing import load_data
from summarization import summarize_multiple
from pathlib import Path
import os


def run_and_save(n_iter=250, length=4, capacity=.1, summary_path="summary", targets_path="targets"):
    train, valid, test = load_data("data")

    summaries = summarize_multiple(test.article, n_iter=n_iter, length=length, capacity=capacity)
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
