import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import tabulate
import os

from tool.file_and_directory_management import read_file_to_list, save_to_pickle, load_from_pickle
from tool.data_generator import data_from_json


def organize_entities(entities_gold, entities_matcher,
                      sentences, debug_mode=False):
    gold = []
    matcher = []
    sentence_errors = []

    for sent_id, (sent_gold_entities, sent_matcher_entities, sentence) in enumerate(
            zip(entities_gold, entities_matcher, sentences)):
        sent_gold = []
        sent_matcher = []
        sent_json = {
            'false_positive': [],
            'false_negative': [],
            'text': sentence}

        for gold_entity in sent_gold_entities:
            sent_gold.append(gold_entity[2])
            if gold_entity in sent_matcher_entities:  # if there is the same entity in predictions -> TP
                sent_matcher.append(gold_entity[2])

            else:
                # if there isn't the same entity in predictions -> FN
                sent_matcher.append('-')
                sent_json['false_negative'].append(
                    sentence[gold_entity[0]:gold_entity[1]])

        for matcher_entity in sent_matcher_entities:
            if matcher_entity not in sent_gold_entities:  # if there isn't the same entity in goldstandard -> FP
                sent_gold.append('-')
                sent_matcher.append(matcher_entity[2])
                sent_json['false_positive'].append(
                    sentence[matcher_entity[0]:matcher_entity[1]])

        gold.extend(sent_gold)
        matcher.extend(sent_matcher)
        if debug_mode and (sent_json['false_positive']
                           or sent_json['false_negative']):
            sentence_errors.append(sent_json)
            print(sent_id, sent_json['text'])
            print('not recognized', sent_json['false_negative'])
            print('wrongly recognized', sent_json['false_positive'])
            print()

    return gold, matcher, sentence_errors


def calculate_metrics(gold, matcher, protagonist_tagger=False):
    characters = list(set(gold + matcher))
    if '-' in characters:
        characters.remove('-')

    if protagonist_tagger:
        result = precision_recall_fscore_support(
            np.array(gold),
            np.array(matcher),
            labels=characters,
            average='micro')
    else:
        result = precision_recall_fscore_support(
            np.array(gold),
            np.array(matcher),
            labels=characters)

    result = list(result)
    result[0:3] = [np.round(a, 3) for a in result[0:3]]

    return result


def compute_overall_stats(titles, gold_standard_path,
                          prediction_path, stats_path, protagonist_tagger=False, debug_mode=False):
    gold_overall = []
    matcher_overall = []

    for title in titles:
        entities_gold, sentences = data_from_json(
            os.path.join(gold_standard_path, title + '.json'))
        entities_matcher, _ = data_from_json(
            os.path.join(prediction_path, title + '.json'))

        entities_gold = [[list(x) for x in set(tuple(x) for x in sent_gold_entities)] for sent_gold_entities in
                         entities_gold]
        gold, matcher, errors = organize_entities(
            entities_gold, entities_matcher, sentences, debug_mode)
        metrics_title = calculate_metrics(gold, matcher, protagonist_tagger)
        save_to_pickle(metrics_title, os.path.join(stats_path, title))

        gold_overall.extend(gold)
        matcher_overall.extend(matcher)

    metrics_overall = calculate_metrics(
        gold_overall, matcher_overall, protagonist_tagger)
    save_to_pickle(
        metrics_overall,
        os.path.join(
            stats_path,
            "overall_metrics"))
    return metrics_overall


def metrics(titles_path, gold_standard_path, prediction_path, stats_path,
            protagonist_tagger=False, print_results=False, debug_mode=False):
    titles = read_file_to_list(titles_path)

    compute_overall_stats(
        titles,
        gold_standard_path,
        prediction_path,
        stats_path,
        protagonist_tagger=protagonist_tagger,
        debug_mode=debug_mode)

    if print_results:
        results = get_results(stats_path, titles)
        print(results)


def get_results(stats_path, titles):
    metrics_table = []
    headers = ["Novel title", "Precision", "Recall", "F-measure"]

    for title in titles:
        metrics_title = load_from_pickle(os.path.join(stats_path, title))
        metrics_table.append([title].__add__([m for m in metrics_title]))

    metrics_overall = load_from_pickle(
        os.path.join(stats_path, 'overall_metrics'))
    metrics_table.append(
        ["*** overall results ***"].__add__([m for m in metrics_overall]))
    return tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex')
