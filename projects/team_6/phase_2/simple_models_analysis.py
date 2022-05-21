import pandas as pd
from models.naive_bayes import HateSpeechNaiveBayes
from models.svm import HateSpeechSVM_Rbf, HateSpeechSVM_Poly, HateSpeechSVM_Linear
from data_loading import load_full_data
from sklearn.metrics import f1_score

data = load_full_data(stem=True)
results = []
for model in [HateSpeechSVM_Rbf, HateSpeechSVM_Poly, HateSpeechSVM_Linear, HateSpeechNaiveBayes]:
    for ngram in [(1, 1), (1, 2)]:
        for selector in [5, 10]:
            lmodel = model(ngram_range=ngram, selector_percentile=selector)
            print(f'Running {lmodel.label}')
            lmodel.fit(data=data['train']['data'], labels=data['train']['labels'])
            predictions = lmodel.predict(data=data['test']['data'])
            f1_result_macro = f1_score(data['test']['labels'], predictions, average='macro')
            f1_result_micro = f1_score(data['test']['labels'], predictions, average='micro')
            result = {"label": lmodel.label, "macro": f1_result_macro, "micro": f1_result_micro}
            results.append(result)


result_df = pd.DataFrame(results)
result_df.to_csv("results.csv", sep="\t", index=False)
