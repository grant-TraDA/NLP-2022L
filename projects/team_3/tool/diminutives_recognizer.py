import collections
import csv
import os
# https://github.com/carltonnorthern/nickname-and-diminutive-names-lookup/blob/master/names.csv
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIMINUTIVES_FILE = os.path.join(
    ROOT_DIR, "additional_resources/diminutives.csv")


def create_diminutives_dictionary():
    diminutives_dictionary = collections.defaultdict(list)
    with open(DIMINUTIVES_FILE) as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            names = set(line)
            for name in names:
                diminutives_dictionary[name].append(names)

    return diminutives_dictionary


# returns all names and diminutives associated with the given diminutive
def get_names_from_diminutive(diminutive):
    diminutives_dictionary = create_diminutives_dictionary()
    diminutive = diminutive.lower()
    if diminutive not in diminutives_dictionary:
        return None

    names = set().union(*diminutives_dictionary[diminutive])
    if diminutive in names:
        names.remove(diminutive)

    return names
