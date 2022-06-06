import re
import sys
import glob
import os.path
import random
import json
from typing import List, Tuple


def __load_json(path: str) -> Tuple[str]:
    data = json.load(open(path, encoding="utf-8"))

    text = ""
    text_in = ""
    text_exp = ""
    for token in data["words"]:
        text += token["word"] + token["punctuation"]
        if re.match('^[^\\w"%]+$', token["word"]):
            pass

        else:
            text_in += token["word"]

            if token["punctuation"] == "-" and token["space_after"] is False:
                text_exp += token["word"]
            else:
                text_exp += token["word"] + token["punctuation"]

        if token["space_after"] or token["punctuation"] != "":
            text_in += " "

        if token["space_after"] or token["punctuation"] != "":

            text_exp += " "
        text += " "

    text_in = text_in.lower()
    text_in = re.sub("[,!?.:;-]", " ", text_in)
    text = text.lower()
    text = re.sub("(\\? )+", "? ", text)
    text = re.sub("(! )+", "! ", text)

    text_exp = text_exp.lower()
    text_exp = re.sub(r" ([,!?.:;-])", "\\1", text_exp)
    text_exp = re.sub(r"[,!?.:;-]([^ ])", " \\1", text_exp)
    text_exp = re.sub(r" [,!?.:;-] ", " ", text_exp)

    text_in = text_exp
    text_in = re.sub("([^ ])[,!?.:;-]( |$)", "\\1 ", text_in)

    try:
        assert len(text_in.strip().split(" ")) == len(text_exp.strip().split(" "))
    except AssertionError:
        print(
            len(text_in.strip().split(" ")),
            len(text_exp.strip().split(" ")),
            file=sys.stderr,
        )

        print(text_exp.strip().split(" "), file=sys.stderr)
        print(text_in.strip().split(" "), file=sys.stderr)
        for a, b in zip(text_exp.strip().split(" "), text_in.strip().split(" ")):
            print(a, b, file=sys.stderr)
        print()

    return text_in.strip(), text_exp.strip()


def __read_names(path: str) -> List[str]:
    names = []
    for in_line in open(path, encoding="utf-8"):
        if in_line[-1] == "\n":
            in_line = in_line[:-1]
        name, text = in_line.split("\t")
        names.append(name)
    return names


def parse_data(
        train_path: str, test_path: str, data: List[str], save_path: str
        ) -> None:
    train_names = __read_names(train_path)
    test_names = __read_names(test_path)

    train_paths = []
    test_paths = []
    rest_paths = []
    for path in data:
        json_paths = glob.glob(path + "/*.json")
        for json_path in json_paths:
            basename = os.path.basename(json_path).split(".")[0]
            if basename in train_names:
                train_paths.append(json_path)
            elif basename in test_names:
                test_paths.append(json_path)
            else:
                rest_paths.append(json_path)

    test_paths2 = []
    for name in test_names:
        for path in test_paths:
            if name in path:
                test_paths2.append(path)
                break
    test_paths = test_paths2

    train_paths2 = []
    for name in train_names:
        for path in train_paths:
            if name in path:
                train_paths2.append(path)
                break
    train_paths = train_paths2

    random.seed(0)
    random.shuffle(rest_paths)

    for name, paths in [
        ("test", test_paths),
        ("train", train_paths),
        ("rest", rest_paths),
    ]:
        with open(
            save_path + f"/{name}_expected.tsv", "w", encoding="utf-8"
        ) as out_expected, open(
            save_path + f"/{name}_in.tsv", "w", encoding="utf-8"
        ) as out_in:
            for path in paths:
                json_in, json_expected = __load_json(path)

                basename = os.path.basename(path).split(".")[0]

                out_in.write(f"{basename}\t{json_in}\n")
                out_expected.write(f"{json_expected}\n")
