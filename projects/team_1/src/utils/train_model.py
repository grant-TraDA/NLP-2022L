import pandas as pd
import sklearn
import os
from simpletransformers.ner import NERArgs, NERModel
import torch
import time


labels = ["B", ":", ";", ",", ".", "-", "?", "!"]


def __merge_data(func, y_true, y_pred, **kwargs):
    y_true_res = []
    y_pred_res = []

    for t in y_true:
        y_true_res.extend(t)

    for p in y_pred:
        y_pred_res.extend(p)

    return func(y_true_res, y_pred_res, **kwargs)


def __f1_per_label(y_true, y_pred):
    values = __merge_data(
        sklearn.metrics.f1_score,
        y_true,
        y_pred,
        labels=labels[1:],
        average=None,
        zero_division=0,
    )
    return {str(i): v for i, v in enumerate(values)}


def __pr_per_label(y_true, y_pred):
    values = __merge_data(
        sklearn.metrics.precision_score,
        y_true,
        y_pred,
        labels=labels[1:],
        average=None,
        zero_division=0,
    )
    return {str(i): v for i, v in enumerate(values)}


def __rc_per_label(y_true, y_pred):
    values = __merge_data(
        sklearn.metrics.recall_score,
        y_true,
        y_pred,
        labels=labels[1:],
        average=None,
        zero_division=0,
    )
    return {str(i): v for i, v in enumerate(values)}


def train_model(
    train_data_dir,
    eval_data_dir,
    test_data_dir,
    save_dirpath,
    epochs=5,
    learning_rate=5e-5,
    batch_size=32,
    grad_acc_steps=1,
    model_name="allegro/herbert-base-cased",
    model_type="herbert",
    warmup_steps=1,
    eval_steps=100,
    max_seq_len=256,
    focal_alpha=0.25,
    seed=81945,
    early_stopping_metric="f1_weighted",
    eval_during_training=True,
    weights=None,
    use_dice=True,
    use_focal=True,
):

    train_data = pd.read_csv(train_data_dir, sep="\t", header=0)
    eval_data = pd.read_csv(eval_data_dir, sep="\t", header=0)

    sentences_to_delete = train_data[
        ~train_data.labels.isin(labels)
    ].sentence_id
    train_data = train_data.loc[
        ~train_data.sentence_id.isin(sentences_to_delete)
        ]

    sentences_to_delete = eval_data[~eval_data.labels.isin(labels)].sentence_id
    eval_data = eval_data.loc[~eval_data.sentence_id.isin(sentences_to_delete)]

    ner_args = NERArgs()
    ner_args.early_stopping_metric = early_stopping_metric
    ner_args.early_stopping_metric_minimize = False
    ner_args.model_type = model_type
    ner_args.model_name = model_name
    ner_args.train_batch_size = batch_size
    ner_args.eval_batch_size = batch_size
    ner_args.gradient_accumulation_steps = grad_acc_steps
    ner_args.learning_rate = learning_rate
    ner_args.num_train_epochs = epochs
    ner_args.evaluate_during_training = eval_during_training
    ner_args.evaluate_during_training_steps = eval_steps
    ner_args.max_seq_length = max_seq_len
    ner_args.manual_seed = seed
    ner_args.warmup_steps = warmup_steps
    ner_args.save_eval_checkpoints = False
    ner_args.use_multiprocessing = False
    ner_args.use_multiprocessing_for_evaluation = False

    if use_dice:
        ner_args.loss_type = "dice"
        ner_args.loss_args = {
            "smooth": 0.001,
            "square_denominator": True,
            "with_logits": True,
            "ohem_ratio": 0.0,
            "alpha": 0,
            "reduction": "mean",
            "index_label_position": True,
        }
    if use_focal:
        ner_args.loss_type = "focal"
        ner_args.loss_args = {
            "alpha": focal_alpha,
            "gamma": 2,
            "reduction": "mean",
            "ignore_index": -100,
        }

    metrics = {
        "f1_micro": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.f1_score,
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
            labels=labels[1:],
        ),
        "f1_macro": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.f1_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
            labels=labels[1:],
        ),
        "f1_weighted": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.f1_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            labels=labels[1:],
        ),
        "pr_micro": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.precision_score,
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
            labels=labels[1:],
        ),
        "pr_macro": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.precision_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
            labels=labels[1:],
        ),
        "pr_weighted": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.precision_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            labels=labels[1:],
        ),
        "rc_micro": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.recall_score,
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
            labels=labels[1:],
        ),
        "rc_macro": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.recall_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
            labels=labels[1:],
        ),
        "rc_weighted": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.recall_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            labels=labels[1:],
        ),
        "confusion_matrix": lambda y_true, y_pred: __merge_data(
            sklearn.metrics.confusion_matrix, y_true, y_pred, labels=labels[1:]
        ),
        "f1_class": lambda y_true, y_pred: __f1_per_label(y_true, y_pred),
        "pr_class": lambda y_true, y_pred: __pr_per_label(y_true, y_pred),
        "rc_class": lambda y_true, y_pred: __rc_per_label(y_true, y_pred),
    }

    train_data.words = train_data.words.astype(str)
    eval_data.words = eval_data.words.astype(str)
    print("labels", labels)
    start = time.time()
    output_dir = f"{save_dirpath}/model_dir_{model_name}_{start}"

    ner_args.output_dir = output_dir
    ner_args.best_model_dir = os.path.join(output_dir, "best_model")
    model = NERModel(
        model_type,
        model_name,
        labels=labels,
        args=ner_args,
        weight=weights,
        use_cuda=True if torch.cuda.is_available() else False,
    )

    model.train_model(
        train_data, output_dir=output_dir, eval_data=eval_data, **metrics
        )
    return ner_args.best_model_dir
