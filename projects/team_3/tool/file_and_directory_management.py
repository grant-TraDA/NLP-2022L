import errno
import os
import os.path
import argparse
from tabulate import tabulate
import pickle


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def open_path(path, mode):
    mkdir(os.path.dirname(path))
    return open(path, mode, encoding="utf-8")


def read_file_to_list(path):
    file = open_path(path, "r")
    lines = file.readlines()
    strings = []

    for line in lines:
        line = line.encode('ascii', 'ignore').decode("utf-8")
        strings.append(line.rstrip())

    return strings


def read_file(path):
    file = open_path(path, "r")
    text = file.read()

    return text.encode('ascii', 'ignore').decode("utf-8")


def read_sentences_from_file(path):
    file = open_path(path, "r")
    text = file.readlines()
    for index, sentence in enumerate(text):
        text[index] = sentence.replace('\n', '')

    return text


def write_list_to_file(path, list_to_write):
    file = open_path(path, "w+")
    file.write(tabulate(list_to_write, tablefmt='orgtbl'))
    file.close()


def write_text_to_file(path, text):
    file = open_path(path, "w+")
    file.write(text)
    file.close()


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"{path} is not a valid directory path")


def file_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file path")


def save_to_pickle(data, path):
    mkdir(os.path.dirname(path))
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def load_from_pickle(path):
    file = open(path, "rb")
    data = pickle.load(file)
    return data
