from deep_learning.embedders import TfidfEmbedder, CountEmbedder, SpacyEmbedder
from deep_learning.mlp import MLP
from data_loading import load_full_data
from deep_learning.dataset import CustomDataset
import pandas as pd
from torch import nn
import torch
from sklearn.metrics import f1_score
import random
import matplotlib.pyplot as plt
import os
import json

random.seed(1)


def subsample_data(data):
    datas = []
    labels = []
    for x, y in zip(data['train']['data'], data['train']['labels']):
        if random.random() < 0.8 and int(y) in [0]:
            continue
        else:
            datas.append(x)
            labels.append(y)
    df_train = pd.DataFrame({"sentence": datas, "target": labels})
    df_train = df_train.sample(frac=1)
    return df_train


def train_mlp(train_ds, test_ds, net, epochs=20):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    results = []
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}')

        for i, data in enumerate(train_ds, 0):
            # print(f"Running {i} processing of {epoch + 1} epoch")
            inputs, targets = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
        for i, data in enumerate(test_ds):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.argmax(1)
            targets = targets.argmax(1)
            f1_result_macro = f1_score(outputs, targets, average='macro')
            f1_result_micro = f1_score(outputs, targets, average='micro')
            curr_result = {
                "epoch": epoch+1,
                "f1_macro": f1_result_macro,
                "f1_micro": f1_result_micro
            }
            print(curr_result)
            results.append(curr_result)
    return results


def experiment(embedder, stem=False, epochs=20):
    random.seed(1)
    data = load_full_data(stem=stem)
    embedder.fit(data['train']['data'], data['train']['labels'])

    experiment_label = f"mlp-{embedder.label}-stem={str(stem).lower()}"
    print(f"Running experiment {experiment_label}")

    net = MLP(embedder.feature_count)

    df_train = subsample_data(data)
    ds_train = CustomDataset(df_train, embedder)
    dataset_train = torch.utils.data.DataLoader(ds_train, batch_size=10)

    df_test = pd.DataFrame({"sentence": data['test']['data'], "target": data['test']['labels']})
    ds_test = CustomDataset(df_test, embedder)
    dataset_test = torch.utils.data.DataLoader(ds_test, batch_size=1000)

    results = train_mlp(dataset_train, dataset_test, net, epochs)

    results_path = os.path.join(os.path.dirname(__file__), "results")
    result_path = os.path.join(results_path, experiment_label)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    json_results_path = os.path.join(result_path, "result.json")
    png_plot_path = os.path.join(result_path, "plot.png")
    with open(json_results_path, 'w+') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

    y = []
    x = []
    for idx, result in enumerate(results):
        y.append(result['f1_macro'])
        x.append(idx+1)
    plt.ylim(0, 1)
    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.ylabel("F1 macro")
    plt.title(f"MLP + {embedder.label}")
    plt.savefig(png_plot_path)
    plt.clf()
    plt.cla()
    plt.close()
    return {"results": results, "label": experiment_label}


embedders = [
    TfidfEmbedder(),
             CountEmbedder(ngram_range=(1, 1), selector_percentile=5),
             CountEmbedder(ngram_range=(1, 2), selector_percentile=5),
             SpacyEmbedder(corpus_type="pl_core_news_md"),
             SpacyEmbedder(corpus_type="pl_core_news_lg")
             ]

epochs = 20
results = []
for stem in [
    True,
    False
]:
    for embedder in embedders:
        result = experiment(embedder, stem=stem, epochs=epochs)
        print(result)
        results.append(result)


with open("results.txt", 'w+') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=4)


plt.figure(figsize=(12, 6), dpi=80)
plt.ylim(0, 1)
plt.xticks([1, 5, 10, 15, 20])
xs = [i+1 for i in range(epochs)]
for result in results:
    ys = []
    label = result['label']
    for x in result['results']:
        ys.append(x['f1_macro'])
    plt.plot(xs, ys, label=label)

plt.xlabel('epoch')
plt.ylabel("F1 macro")
plt.title('MLP with different embeddings')
plt.legend(bbox_to_anchor=(0, -0.1), loc="upper left")
plt.savefig("mlp_with_different_embeddings_comparison.png", bbox_inches="tight")

plt.clf()
plt.cla()
plt.close()
