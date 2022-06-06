import pandas as pd
import numpy as np
import random


def convert_to_pandas(
    data_file: str,
    out_file: str,
    train_to_test_dev_ratio: float = 0.8,
    dev_to_test_ratio: float = 0.5,
    seed: int = 1353,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    words = [[]]
    tags = [[]]
    spaces_after = [[]]

    with open(data_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("-DOCSTART-"):
                continue
            if len(line.strip()) == 0:
                words.append([])
                tags.append([])
                spaces_after.append([])
                continue

            try:
                word, tag, space = line.strip().split("\t")
            except Exception:
                try:
                    word, tag = line.strip().split("\t")
                    space = " "
                except Exception:
                    continue

            words[-1].append(word)
            tags[-1].append(tag)
            spaces_after[-1].append(space)

        if not words[-1]:
            words = words[:-1]
            tags = tags[:-1]
            spaces_after = spaces_after[:-1]

    data = []
    for example_id, (example_words, example_tags, space_after) in enumerate(
        zip(words, tags, spaces_after)
    ):
        example_data = pd.DataFrame(
            {
                "words": example_words,
                "labels": example_tags,
                "times": space_after,
                "sentence_id": example_id,
            }
        )
        data.append(example_data)
    data = pd.concat(data)

    if out_file:
        data.to_csv(out_file, sep="\t")
    else:
        data.to_csv("data.tsv", sep="\t")
