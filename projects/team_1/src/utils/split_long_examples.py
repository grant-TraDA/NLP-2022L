import pandas as pd
from transformers import AutoTokenizer


def __split_long_examples(
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 256,
        stride: float = 0.8) -> pd.DataFrame:

    splitted_data = []

    for sentence_id, example in data.groupby("sentence_id"):
        words_with_labels = []
        words_in_example = 0
        tokenized_len = 0
        token_lens = []
        chunk_id = 0

        for word, label, space in zip(
                example.words, example.labels, example.times
                ):
            tokenized_word = tokenizer.tokenize(word)
            if tokenized_len + len(tokenized_word) >= max_seq_len - 1:
                splitted_data.extend(
                    [
                        (w, l, s, f"{sentence_id}_{chunk_id}")
                        for w, l, s in words_with_labels
                    ]
                )
                chunk_id += 1
                offset = int(words_in_example * stride)
                words_with_labels = words_with_labels[offset:]
                tokenized_len -= sum(token_lens[:offset])
                token_lens = token_lens[offset:]
                words_in_example -= offset

            token_lens.append(len(tokenized_word))
            tokenized_len += len(tokenized_word)
            words_with_labels.append((word, label, space))
            words_in_example += 1

        if tokenized_len >= 0:
            splitted_data.extend(
                [
                    (w, l, s, f"{sentence_id}_{chunk_id}")
                    for w, l, s in words_with_labels
                ]
            )

    return pd.DataFrame(
        splitted_data, columns=["words", "labels", "times", "sentence_id"]
    )


def split_long_examples(
        data_path: str,
        out_file: str,
        max_seq_len: int = 256,
        stride: int = 1,
        tokenizer_path: str = "allegro/herbert-base-cased",
        ) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data = pd.read_csv(
        data_path,
        sep="\t",
        keep_default_na=False,
        dtype={"words": "str", "labels": "str", "times": "str"},
    )

    data = __split_long_examples(
        data, tokenizer, max_seq_len=max_seq_len, stride=stride
    )
    data.to_csv(out_file, sep="\t")
