from sklearn.feature_extraction.text import CountVectorizer
from data_loading import load_full_data
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os
import json


RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results/")
TREE_DEPTH_TEST_PATH = os.path.join(RESULTS_PATH, "tree-depth-test")
if not os.path.exists(TREE_DEPTH_TEST_PATH):
    os.makedirs(TREE_DEPTH_TEST_PATH)
RESULT_PATH = os.path.join(TREE_DEPTH_TEST_PATH, "result.json")
PLOT_PATH = os.path.join(TREE_DEPTH_TEST_PATH, "plot.png")

max_depth = 500
step = 20


def experiment():
    data = load_full_data(stem=False)
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    vectorizer.fit(data['train']['data'])
    train_embeddings = vectorizer.transform(data['train']['data'])

    y = []
    x = []

    results = []

    for depth in [1] + list(range(step, max_depth+1, step)):
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(train_embeddings, data['train']['labels'])
        test_embeddings = vectorizer.transform(data['test']['data'])
        result = clf.predict(test_embeddings)
        f1_result_macro = f1_score(data['test']['labels'], result, average='macro')
        f1_result_micro = f1_score(data['test']['labels'], result, average='micro')
        x.append(depth)
        y.append(f1_result_macro)
        result = {"depth": depth, "f1_macro": f1_result_macro, "f1_micro": f1_result_micro}
        results.append(result)
        print(result)

    plt.figure(figsize=(14, 6), dpi=80)
    plt.ylim(0, 1)
    plt.xticks([1] + [d for d in range(step, max_depth+1, step)])

    plt.plot(x, y)
    plt.xlabel('tree depth')
    plt.ylabel("F1 macro")
    plt.title('Decision tree, height test')
    plt.savefig(PLOT_PATH)

    plt.show()

    with open(RESULT_PATH, 'w+') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)


def plot_combined_plot():
    path1 = os.path.join(RESULTS_PATH, "tree-depth-test-stem=false")
    path2 = os.path.join(RESULTS_PATH, "tree-depth-test-stem=true")

    nostem = os.path.join(path1, "result.json")
    stem = os.path.join(path2, "result.json")

    with open(nostem, 'r') as fp:
        nostem = json.load(fp)

    with open(stem, 'r') as fp:
        stem = json.load(fp)

    stem_x = [x['depth'] for x in stem]
    nostem_x = [x['depth'] for x in nostem]
    stem_y = [x['f1_macro'] for x in stem]
    nostem_y = [x['f1_macro'] for x in nostem]

    plt.figure(figsize=(12, 6), dpi=80)
    plt.ylim(0, 1)
    plt.xticks([1] + [d for d in range(step, max_depth + 1, step)])

    plt.plot(stem_x, stem_y, label='with stemmer')
    plt.plot(nostem_x, nostem_y, label='without stemmer')
    plt.xlabel('tree depth')
    plt.ylabel("F1 macro")
    plt.title('Decision tree, height test')
    plt.legend()

    plt.savefig(PLOT_PATH)

    plt.show()


def plot_decision_tree():
    data = load_full_data(stem=False)
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    vectorizer.fit(data['train']['data'])
    train_embeddings = vectorizer.transform(data['train']['data'])
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(train_embeddings, data['train']['labels'])
    rv = {v: k for k, v in vectorizer.vocabulary_.items()}
    feature_names = [rv[i] for i in range(len(rv))]
    class_names = ['neutral', 'cyberbullying', 'hate']

    plt.figure(figsize=(14, 9), dpi=200)
    plot_tree(clf, feature_names=feature_names, class_names=class_names, fontsize=9)
    plt.show()

    test_embeddings = vectorizer.transform(data['test']['data'])
    result = clf.predict(test_embeddings)
    f1_result_macro = f1_score(data['test']['labels'], result, average='macro')
    f1_result_micro = f1_score(data['test']['labels'], result, average='micro')
    print(f"macro: {f1_result_macro}")
    print(f"micro: {f1_result_micro}")


if __name__ == '__main__':
    #experiment()
    # plot_combined_plot()
    plot_decision_tree()
