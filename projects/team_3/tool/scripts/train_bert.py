from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForTokenClassification
import numpy as np
from datasets import load_metric
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
import pandas as pd
from huggingface_hub import notebook_login
from transformers import AutoTokenizer

train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')

train.tokens = [eval(x) for x in train.tokens]
train.ner_tags = [eval(x) for x in train.ner_tags]
valid.tokens = [eval(x) for x in valid.tokens]
valid.ner_tags = [eval(x) for x in valid.ner_tags]

dataset = DatasetDict({'train': Dataset.from_pandas(
    train), 'valid': Dataset.from_pandas(valid)})

label_names = [
    'O',
    'B-PER',
    'I-PER',
    'B-ORG',
    'I-ORG',
    'B-LOC',
    'I-LOC',
    'B-MISC',
    'I-MISC']


model_checkpoint = "Davlan/bert-base-multilingual-cased-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

inputs = tokenizer(dataset["train"][0]["tokens"], is_split_into_words=True)
labels = dataset["train"][0]["ner_tags"]
word_ids = inputs.word_ids()


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
metric = load_metric("seqeval")
labels = [label_names[i] for i in labels]


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(
        predictions=true_predictions,
        references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)


args = TrainingArguments(
    "bert-finetuned-protagonist",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()

trainer.push_to_hub(commit_message="Training complete")
