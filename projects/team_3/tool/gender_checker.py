import collections
import csv
import os
import numpy as np
import gender_guesser.detector as gender_guesser

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GENDER_FILE = os.path.join(ROOT_DIR, "additional_resources/gender_markers.csv")


def create_titles_and_gender_dictionary():
    gender_dictionary = collections.defaultdict(list)
    with open(GENDER_FILE) as file:
        csv_reader = csv.reader(file, delimiter=' ')
        for line in csv_reader:
            gender_dictionary[line[0]].append(line[1])

    return gender_dictionary


def get_personal_titles():
    titles = []
    with open(GENDER_FILE) as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            titles.append(line[0].split(' ')[0])

    return np.array(titles)


def get_name_gender(name):
    name_elements = name.split()

    titles_gender = create_titles_and_gender_dictionary()
    for element in name_elements:
        if element.replace(".", "") in titles_gender.keys():
            return str(titles_gender[element.replace(".", "")][0])

    gender_detector = gender_guesser.Detector()
    gender = gender_detector.get_gender(name_elements[0])
    if gender == "andy" or gender == "unknown":
        if len(name_elements) > 1:
            gender = gender_detector.get_gender(name_elements[1])
            if gender == "andy" or gender == "unknown":
                return None
        else:
            return None
    if gender == "female" or gender == "mostly_female":
        return 'f'
    if gender == "male" or gender == "mostly_male":
        return 'm'
