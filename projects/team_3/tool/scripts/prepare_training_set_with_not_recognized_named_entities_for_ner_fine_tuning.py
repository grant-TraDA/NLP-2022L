import argparse
import random
import copy
import json
import os
import spacy
from spacy.tokens import Span

from tool.file_and_directory_management import read_file_to_list, write_text_to_file, read_file
from tool.file_and_directory_management import file_path


# title - title of the novel that is to be used for training set creation (the title should not contain any special
#       characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# not_recognized_named_entities_person_file_path - path to .txt file containing all the named entities of category
#       person not recognized by standard NER model
def get_not_recognized_entities(
        title, not_recognized_named_entities_person_file_path):
    named_entities = read_file_to_list(os.path.join(
        not_recognized_named_entities_person_file_path, title))
    return named_entities


# title - title of the novel that is to be used for training set creation (the title should not contain any special
#       characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# novels_texts_file_path - path to .txt file containing full text of the novel
# named_entities - list of named entities of category person that were not recognized properly by standard NER model
#       and that should be included in the training set
# sentences_per_entity - the upper limit for number of sentences with each entity that should be included in the
#       training set
# training_set_dir - path to the directory where the generated (not
# annotated yet) training set should be saved
def create_ner_training_set(title, novels_texts_file_path,
                            named_entities, sentences_per_entity, training_set_dir):
    nlp = spacy.load("en_core_web_sm")
    novel_text = read_file(os.path.join(novels_texts_file_path, title))
    doc = nlp(novel_text)
    potential_sentences = {}
    temporal_named_entities = copy.copy(named_entities)

    for entity in named_entities:
        potential_sentences[entity] = list()

    for sentence in doc.sents:
        if len(temporal_named_entities) == 0:
            break
        for entity in temporal_named_entities:
            if entity in sentence.text:
                current_list = potential_sentences[entity]
                current_list.append(sentence.text)
                potential_sentences[entity] = current_list
                if len(current_list) > sentences_per_entity:
                    temporal_named_entities.remove(entity)
                break

    selected_sentences = []
    for entity_potential_sentences in [
            potential_sentences[entity] for entity in named_entities]:
        count = min(sentences_per_entity, len(entity_potential_sentences))
        selected_sentences.extend(
            [sent for sent in random.sample(entity_potential_sentences, k=count)])

    test_sample = "\n".join(selected_sentences)
    write_text_to_file(
        os.path.join(
            training_set_dir,
            "not_annotated_training_set",
            title),
        test_sample)

    return selected_sentences


def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def overlap(intervals, interval):
    for inter in intervals:
        if get_overlap(inter, interval) > 0:
            return True
    return False


def annotate_training_set(data, names_entities):
    nlp = spacy.load("en_core_web_sm")
    label = "PERSON"
    result = []
    for sentence in data:
        doc = nlp(sentence)
        sent_dict = {}
        entities = []
        intervals = []

        for entity in names_entities:
            if entity in sentence:
                start_index = sentence.index(entity)
                end_index = start_index + len(entity)
                if not overlap(intervals, [start_index, end_index]):
                    entities.append([start_index, end_index, label])
                    intervals.append([start_index, end_index])

        for index, ent in enumerate(doc.ents):
            span = Span(doc, ent.start, ent.end, label=ent.label_)
            doc.ents = [span if e == ent else e for e in doc.ents]

            if not overlap(intervals, [ent.start_char, ent.end_char]):
                entities.append([ent.start_char, ent.end_char, ent.label_])

        sent_dict["content"] = doc.text
        sent_dict["entities"] = list(entities)
        result.append(sent_dict)

    return result


# titles_path - path to .txt file with titles of novels from which the sampled data are to be generated (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# not_recognized_named_entities_person_file_path - path to .txt file containing all the named entities of category
#       person not recognized by standard NER model
# novels_texts_file_path - path to .txt file containing full text of the novel
# sentences_per_entity - the upper limit for number of sentences with each entity that should be included in the
#       training set
# training_set_dir - path to the directory where the generated training
# set should be saved
def main(titles_path, not_recognized_named_entities_person_file_path, novels_texts_file_path,
         sentences_per_entity, training_set_dir):
    titles = read_file_to_list(titles_path)
    for title in titles:
        named_entities = get_not_recognized_entities(
            title, not_recognized_named_entities_person_file_path)
        ner_training_set = create_ner_training_set(title, novels_texts_file_path, named_entities,
                                                   sentences_per_entity, training_set_dir)
        training_set = annotate_training_set(ner_training_set, named_entities)
        with open(os.path.join(training_set_dir, title), 'w') as result:
            json.dump(training_set, result)
        print("One novel done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path)
    parser.add_argument(
        'not_recognized_named_entities_person_file_path',
        type=file_path)
    parser.add_argument('novels_texts_file_path', type=file_path)
    parser.add_argument('sentences_per_entity', type=int)
    parser.add_argument('training_set_dir', type=str)
    opt = parser.parse_args()
    main(opt.titles_path, opt.not_recognized_named_entities_person_file_path, opt.novels_texts_file_path,
         opt.sentences_per_entity, opt.training_set_dir)
