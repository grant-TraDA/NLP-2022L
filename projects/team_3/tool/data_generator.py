import os
import random
import spacy
import json
import errno

from tool.file_and_directory_management import write_text_to_file, read_file
from tool.wiki_scanner import get_descriptions_of_characters, get_list_of_characters

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# extract lists of characters for novels from corresponding articles on Wikipedia (titles should not contain
# any special characters and spaces should be replaced with "_", for
# example "Pride_andPrejudice")
def generate_lists_of_characters(titles):
    for title in titles:
        get_list_of_characters(title)


# extract description of characters for novels from corresponding articles on Wikipedia (titles should not contain
# any special characters and spaces should be replaced with "_", for
# example "Pride_andPrejudice")
def generate_descriptions_of_characters(titles):
    for title in titles:
        get_descriptions_of_characters(title)


# generate sample test data from full novels texts
def generate_sample_test_data(
        titles, number_of_sentences, novels_texts_dir_path, generated_data_dir):
    nlp = spacy.load("en_core_web_sm")

    for title in titles:
        novel_text = read_file(novels_texts_dir_path + title)
        doc = nlp(novel_text)
        potential_sentences = []
        people = set()

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people.add(ent.text)

        for sentence in doc.sents:
            if any(person in sentence.text for person in people):
                potential_sentences.append(sentence.text)

        selected_sentences = [
            sent for sent in random.sample(
                potential_sentences,
                k=number_of_sentences)]
        test_sample = "\n".join(selected_sentences)

        write_text_to_file(generated_data_dir + title, test_sample)


def json_to_spacy_train_data(path):
    with open(path, encoding='utf-8') as train_data:
        train = json.load(train_data)

    train_data = []
    for data in train:
        ents = [tuple(entity) for entity in data['entities']]
        train_data.append((data['content'], {'entities': ents}))

    return train_data


def spacy_format_to_json(path, data, title):
    eval_data = list(eval(data))
    json_data = []

    for sentence in eval_data:
        sent_dict = {
            "content": sentence[0],
            "entities": sentence[1]['entities']}
        json_data.append(sent_dict)

    path = os.path.join(path, title + ".json")

    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(path, 'w+') as result:
        json.dump(json_data, result)


def data_from_json(path):
    with open(path, encoding='utf-8') as train_data:
        train = json.load(train_data)

    entities = []
    contents = []
    for data in train:
        entities.append(data['entities'])
        contents.append(data['content'])

    return entities, contents
