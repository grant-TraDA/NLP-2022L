import os
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")

train = ["Maly_Ksiaze", "Krzyzacy", "Lalka", "Robinson_Crusoe"]
valid = ["Nad_Niemnem", "W_pustyni_i_w_puszczy"]
test = ["Ksiega_dzungli", "Przedwiosnie"]

for dataset, filename in zip([train, valid], ['train.csv', 'valid.csv']):
    df = []
    for book in dataset:
        with open('data/testing_sets/test_person_polish_gold_standard/' + book  + '.json') as f:
            data = json.loads(f.read())
        for sent in data:
            tokens = []
            starts = []
            for w in range(tokenizer(sent['content']).word_ids()[-2]):
                charspan = tokenizer(sent['content']).word_to_chars(w)
                tokens.append(sent['content'][charspan.start:charspan.end])
                starts.append(charspan.start)
            results = []
            ent_id = 0
            i = 0
            entities = sorted(sent['entities'], key=lambda x: x[0])
            while ent_id < len(entities) and i < len(starts):
                if starts[i] < entities[ent_id][0]:
                    results.append(0)
                    i += 1
                elif starts[i] == entities[ent_id][0]:
                    results.append(1)
                    i += 1
                elif starts[i] < entities[ent_id][1]:
                    results.append(2)
                    i += 1
                else:
                    ent_id += 1
            while i < len(starts):
                results.append(0)
                i += 1
            df.append({'tokens': tokens, 'ner_tags': results})
    pd.DataFrame(df).to_csv(filename)