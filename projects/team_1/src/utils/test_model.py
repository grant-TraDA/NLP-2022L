from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import IterableDataset, DataLoader


class TokenizedExamplesDataset(IterableDataset):
    def __init__(self, examples):
        self.examples = examples

    def __iter__(self):
        for example in self.examples:
            yield example

    def __len__(self):
        return len(self.examples)


class ModelForInference:
    def __init__(self, model_path, max_seq_len=256,
                 use_sliding_window=True, stride=0.8,
                 device='cpu', fp16=True, batch_size=1):
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path
            )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.use_sliding_window = use_sliding_window
        self.stride = stride
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        if device == 'cpu':
            fp16 = False

        if device.startswith('cuda'):
            assert torch.cuda.is_available(), "no cuda device detected"

        self.device = device
        self.fp16 = fp16
        self.model.to(self.device)

    def prepare_examples(self, examples, tokenizer):
        tokenized_examples = []
        token_mappings = []
        window_counts = []

        for example in examples:
            tokens = []
            token_id_to_word_id = []
            word_id_to_tokenized_len = []
            num_windows = 0

            for idx, word in enumerate(example):
                tokenized_word = tokenizer.tokenize(word)
                word_id_to_tokenized_len.append(len(tokenized_word))

                if len(tokens) + len(tokenized_word) >= self.max_seq_len - 1:
                    if not self.use_sliding_window:
                        break
                    tokenized_examples.append(
                        tokenizer.convert_tokens_to_ids(
                            [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
                            )
                    )
                    num_windows += 1
                    token_mappings.append([-1] + token_id_to_word_id + [-1])

                    first_word_id = token_id_to_word_id[0]
                    last_word_id = token_id_to_word_id[-1]
                    jump_to_id = first_word_id + int(
                        (last_word_id - first_word_id) * self.stride
                        )
                    offset = sum(
                        word_id_to_tokenized_len[first_word_id:jump_to_id]
                        )

                    tokens = tokens[offset:]
                    token_id_to_word_id = token_id_to_word_id[offset:]

                tokens.extend(tokenized_word)
                token_id_to_word_id.extend([idx] * len(tokenized_word))

            if tokens:
                tokenized_examples.append(
                    tokenizer.convert_tokens_to_ids(
                        [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
                        )
                )
                num_windows += 1
                token_mappings.append([-1] + token_id_to_word_id + [-1])

            window_counts.append(num_windows)

        return (
            TokenizedExamplesDataset(tokenized_examples),
            window_counts,
            token_mappings
        )

    @staticmethod
    def merge_predictions(
            windows_preds, windows_scores, windows_token_mappings
            ):

        result_size = windows_token_mappings[-1][-2] + 1
        results = [0] * result_size
        result_score = [0] * result_size

        prev_word_id = -1

        for window_preds, window_scores, window_token_mappings in \
                zip(windows_preds, windows_scores, windows_token_mappings):
            for pred, score, word_id in zip(
                window_preds, window_scores, window_token_mappings
                    ):
                if word_id == prev_word_id:
                    continue

                prev_word_id = word_id
                if word_id == -1:
                    continue

                if pred != 0:
                    if score > result_score[word_id]:
                        result_score[word_id] = score
                        results[word_id] = pred
        return results

    @staticmethod
    def merge_predictions2(
            windows_preds, windows_scores, windows_token_mappings
                ):
        result_size = windows_token_mappings[-1][-2] + 1
        results = [0] * result_size
        result_score = [-100] * result_size

        prev_word_id = -1

        for window_preds, window_scores, window_token_mappings in \
                zip(windows_preds, windows_scores, windows_token_mappings):
            first_word_id = min([x for x in window_token_mappings if x > 0])
            last_word_id = max(window_token_mappings)

            for pred, score, word_id in zip(
                window_preds, window_scores, window_token_mappings
                    ):
                if word_id == prev_word_id:
                    continue

                prev_word_id = word_id
                if word_id == -1:
                    continue

                if pred != 0:
                    context = min(word_id-first_word_id, last_word_id-word_id)
                    if context >= result_score[word_id]:
                        result_score[word_id] = context
                        results[word_id] = pred
        return results

    def predict(self, to_predict, context):
        tokenizer = self.tokenizer

        if self.fp16:
            from torch.cuda import amp

        dataset, window_counts, token_mappings = self.prepare_examples(
            to_predict, tokenizer
            )

        def collate(examples):
            inputs = tokenizer.pad(
                {"input_ids": examples}, padding="longest", return_tensors="pt"
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            return inputs

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=collate
            )

        preds = []
        scores = []
        for batch in data_loader:
            with torch.no_grad():
                if self.fp16:
                    with amp.autocast():
                        logits = self.model(**batch)[0]
                else:
                    logits = self.model(**batch)[0]

                batch_score, batch_pred = torch.max(logits, dim=2)
                batch_score = batch_score.detach().cpu().numpy()
                batch_pred = batch_pred.detach().cpu().numpy()

                scores.extend(batch_score)
                preds.extend(batch_pred)

        offset = 0

        concated_preds = []
        for window in window_counts:
            if context:
                concated_preds.append(
                    self.merge_predictions2(
                        preds[offset:offset + window],
                        scores[offset:offset + window],
                        token_mappings[offset:offset + window]
                        )
                )
            else:
                concated_preds.append(
                    self.merge_predictions(
                        preds[offset:offset + window],
                        scores[offset:offset + window],
                        token_mappings[offset:offset + window])
                    )
        return concated_preds


def __ids_to_labels(preds, labels):
    return [labels[p] for p in preds]


def test_model(
        path_to_model='best_model',
        path_to_test='test-A/in.tsv',
        path_to_out='test-A/out.tsv'
        ):

    model = ModelForInference(path_to_model)

    pred_labels = ['B', ':', ';', ',', '.', '-', '?', '!']

    with open(path_to_test) as f, open(path_to_out, 'w') as out:
        splitted = [line.split('\t') for line in f]
        for idx, text in splitted:
            preds = __ids_to_labels(
                model.predict([text.split()], context='store_true')[0],
                pred_labels
                )

            for idx, (word, label) in enumerate(zip(text.split(), preds)):
                out.write(
                    f'{"" if idx == 0 else " "}{word}{label if label != "B" else ""}'
                    )
            out.write('\n')
