import argparse
import os
import random
import json
import spacy
from spacy.tokens import Span

from tool.file_and_directory_management import read_file_to_list, write_text_to_file, \
    read_sentences_from_file, read_file
from tool.file_and_directory_management import dir_path, file_path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_NAMES_FILE = os.path.join(
    ROOT_DIR.replace(
        "/scripts",
        ""),
    "additional_resources/common_names.txt")


# common_names_path - path to .txt file containing a list of common English names that should be injected to extend a
#       ner model training set
def get_common_names(common_names_path=COMMON_NAMES_FILE):
    common_names = read_file_to_list(common_names_path)
    return common_names


def get_names_to_be_replaced(characters):
    names_to_be_replaced = [characters[0]]
    if " " in characters[0]:
        names_to_be_replaced.extend(characters[0].split(" "))
    return names_to_be_replaced


# extracting sentences from novels for fine-tuning ner by injecting common names
# titles - titles of novels to be included in the training set
# number_of_sentences - number of to be extracted for the training set
# characters_lists_dir_path - directory of .txt files containing lists of characters from corresponding novels (names of
#       files should be the same as titles on the list from titles_path)
# novels_texts_dir_path - directory of .txt files containing full novels texts (names of files should be the same as
#       titles on the list from titles_path)
# training_set_dir - path to the directory where the generated training
# set should be saved
def extract_sentences_for_names_injection(titles, number_of_sentences, characters_lists_dir_path,
                                          novels_texts_dir_path, training_set_dir):
    sentences_per_novel = number_of_sentences / len(titles)
    nlp = spacy.load("en_core_web_sm")

    for title in titles:
        characters = read_file_to_list(
            os.path.join(characters_lists_dir_path, title))
        novel_text = read_file(os.path.join(novels_texts_dir_path, title))
        names_to_be_replaced = get_names_to_be_replaced(characters)
        doc = nlp(novel_text)
        potential_sentences = []

        for sentence in doc.sents:
            if any(name in sentence.text for name in names_to_be_replaced):
                potential_sentences.append(sentence.text)

        selected_sentences = [sent for sent in random.sample(potential_sentences,
                                                             k=min(int(sentences_per_novel), len(potential_sentences)))]
        test_sample = "\n".join(selected_sentences)

        write_text_to_file(os.path.join(training_set_dir, "extracted_sentences", title),
                           test_sample)


def inject_common_names(common_names, sentences, names_to_be_replaced):
    nlp = spacy.load("en_core_web_sm")
    updated_sentences = []
    result = []

    for sentence in sentences:
        for name in names_to_be_replaced:
            if name in sentence:
                common_name = random.choice(common_names)
                sentence = sentence.replace(name, common_name)

        updated_sentences.append(sentence)
        doc = nlp(sentence)
        sent_dict = {}
        entities = []

        for ent in doc.ents:
            span = Span(doc, ent.start, ent.end, label=ent.label_)
            doc.ents = [span if e == ent else e for e in doc.ents]
            entities.append([ent.start_char, ent.end_char, ent.label_])

        sent_dict["content"] = doc.text
        sent_dict["entities"] = list(entities)
        result.append(sent_dict)

    return result, updated_sentences


# titles_path - path to .txt file with titles of novels from which the sampled data are to be generated (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# characters_lists_dir_path - directory of .txt files containing lists of characters from corresponding novels (names of
#       files should be the same as titles on the list from titles_path)
# novels_texts_dir_path - directory of .txt files containing full novels texts (names of files should be the same as
#       titles on the list from titles_path)
# sentences_per_common_name - the upper limit for number of injections of each common English name from a list that
#       should be included in the training set
# training_set_dir - path to the directory where the generated training
# set should be saved
def main(titles_path, characters_lists_dir_path, novels_texts_dir_path,
         sentences_per_common_name, training_set_dir):
    titles = read_file_to_list(titles_path)
    common_names = get_common_names()
    number_sentences_to_extracted = sentences_per_common_name * \
        len(common_names)
    extract_sentences_for_names_injection(titles, number_sentences_to_extracted, characters_lists_dir_path,
                                          novels_texts_dir_path, training_set_dir)
    sentences = []
    names_to_be_replaced = []
    for title in titles:
        data = read_sentences_from_file(
            os.path.join(
                training_set_dir,
                "extracted_sentences",
                title))
        sentences.extend(data)
        characters = read_file_to_list(
            os.path.join(characters_lists_dir_path, title))
        names_to_be_replaced.extend(get_names_to_be_replaced(characters))

    training_set, updated_sentences = inject_common_names(
        common_names, sentences, names_to_be_replaced)
    with open(os.path.join(training_set_dir, "training_set.json"), 'w+') as result:
        json.dump(training_set, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path)
    parser.add_argument('characters_lists_dir_path', type=dir_path)
    parser.add_argument('novels_texts_dir_path', type=dir_path)
    parser.add_argument('sentences_per_common_name', type=int)
    parser.add_argument('training_set_dir', type=str)
    opt = parser.parse_args()
    main(opt.titles_path, opt.characters_lists_dir_path, opt.novels_texts_dir_path,
         opt.sentences_per_common_name, opt.training_set_dir)
