import json
import pandas as pd
import ast


def annotations_to_pylighter(annotations):
    labels = []
    corpus = []
    for sentence in annotations:
        text = sentence['content']
        corpus.append(text)
        entities = sentence['entities']

        sent_labels = ['O'] * len(text)
        for ent in entities:
            sent_labels[ent[0]] = 'B-' + ent[2]
            for index in range(ent[0] + 1, ent[1]):
                sent_labels[index] = 'I-' + ent[2]
        labels.append(sent_labels)
    return labels, corpus


def csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path, sep=';')
    new_annotations = []
    for sent_id, (text, labels) in enumerate(zip(df.document, df.labels)):
        sent_entities = []
        started = False
        for i, tag in enumerate(ast.literal_eval(labels)):
            if tag.startswith('B'):
                started = True
                entity_type = tag[2:]
                start = i
            if started and tag == 'O':
                started = False
                end = i
                sent_entities.append([start, end, entity_type])
        if started:
            sent_entities.append([start, i + 1, entity_type])
        new_annotations.append({'content': text, 'entities': sent_entities})
    with open(json_path, 'w') as f:
        f.write(json.dumps(new_annotations))
