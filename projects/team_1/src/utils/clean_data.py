# Input data cleaning functions
import os
import re
import json
from functools import reduce


def __rm_consecutive_spaces(string:str) -> str:
    return re.sub(' +', ' ', string)


def __generate_replacement_dict() -> dict:
    # load list of symbols to replace
    symbols_to_replace_infile = open("../data/outputs/eda/symbols_to_replace.txt", "r", encoding="utf-8")
    symbols_to_replace = symbols_to_replace_infile.read().splitlines()
    # load list of noisy words, i.e. words with letters from outside the Polish alphabet
    noisy_words_infile = open("../data/outputs/eda/noisy_words.txt", "r", encoding="utf-8")
    noisy_words = noisy_words_infile.read().splitlines()
    # load list of letters from outside the Polish alphabet
    non_polish_letters_infile = open("../data/outputs/eda/non_polish_letters.txt", "r", encoding="utf-8")
    non_polish_letters = non_polish_letters_infile.read().splitlines()

    # merge noisy data into one list
    symbols_to_replace.extend(noisy_words)
    symbols_to_replace.extend(non_polish_letters)
    symbols_to_replace.extend(["...", "…"])

    # generate dictionary
    replacement_dict = {}

    for symbol in symbols_to_replace:
        replacement_dict[symbol] = ""
    return replacement_dict


def __replace_with_dict(str_to_replace:str, replacement_dict:dict) -> str:
    str_replaced = reduce(lambda x, y: x.replace(*y), [str_to_replace, *list(replacement_dict.items())])
    return str_replaced


def clean_tsv_file(in_path:str, out_path:str) -> None:
    
    target_classes=['.', ',', '?', '!', '-', ':']

    os.makedirs(out_path, mode = 0o777, exist_ok = True) 
    out_path = f"{out_path}/{os.path.basename(in_path)}"

    replacement_dict = __generate_replacement_dict()
    
    if not os.path.exists(out_path):
        open(out_path, 'w+').close()
    
    with open(in_path, encoding="utf-8", mode="r") as f1, open(out_path, encoding="utf-8", mode="w+") as f2:
        for line in f1:
            try:
                name, text = line.split("\t")
            except:
                text = line
                name = None
            text_replaced = __replace_with_dict(text, replacement_dict)
            text_cleaned = __rm_consecutive_spaces(text_replaced)
            for item in target_classes:
                text_cleaned = text_cleaned.replace(f" {item}", item)
            text_cleaned = text_cleaned.lstrip().rstrip()
            if name: line_cleaned = f"{name}\t{text_cleaned}\n"
            else: line_cleaned = f"{text_cleaned}\n"
            

            f2.write(line_cleaned)
            
    f1.close()
    f2.close()
        

def clean_clmtmstmp_file(in_path:str, out_path:str) -> None:
    os.makedirs(out_path, mode = 0o777, exist_ok = True) 
    out_path = f"{out_path}/{os.path.basename(in_path)}"
    
    replacement_dict = __generate_replacement_dict()
    
    with open(in_path, encoding="utf-8", mode="r") as f1, open(out_path, encoding="utf-8", mode="w+") as f2:
        for line in f1:
            try:
                interval, word = line.split(" ")
                word_replaced = __replace_with_dict(word, replacement_dict).strip()
                if word_replaced != "":
                    line_cleaned = f"{interval} {word_replaced}\n"
                else: continue
            except ValueError: line_cleaned = line  # EOF case
            f2.write(line_cleaned)
    f1.close()
    f2.close()

            
def clean_json_file(in_path:str, out_path:str) -> None:
    os.makedirs(out_path, mode = 0o777, exist_ok = True) 
    out_path = f"{out_path}/{os.path.basename(in_path)}"

    replacement_dict = __generate_replacement_dict()
    clean_rows = []

    with open(in_path, encoding="utf-8", mode="r") as f1, open(out_path, encoding="utf-8", mode="w+") as f2:
        data = json.load(f1)

        for row in data['words']:
            new_row = {
                'word': __replace_with_dict(row['word'], replacement_dict).strip(),
                'punctuation': __replace_with_dict(row['punctuation'], replacement_dict).strip(),
                'space_after': row['space_after']
            }
            clean_rows.append(new_row)

        data_clean = {
            'title': data['title'],
            'words': clean_rows
        }

        json.dump(data_clean, f2)
    
    f1.close()
    f2.close()