from fuzzywuzzy import fuzz
import itertools

from tool.gender_checker import get_name_gender, get_personal_titles, create_titles_and_gender_dictionary
from tool.diminutives_recognizer import get_names_from_diminutive
from tool.model.utils import load_model


class NamesMatcher:
    def __init__(self, partial_ratio_precision, library="spacy",
                 model_path="en_core_web_sm", fix_personal_titles=False):
        self.personal_titles = get_personal_titles()
        self.titles_gender_dict = create_titles_and_gender_dictionary()
        self.model = load_model(library, model_path, True, fix_personal_titles)
        self.partial_ratio_precision = partial_ratio_precision

    def recognize_person_entities(self, test_data, characters, full_text):

        ner_results = self.model.get_ner_results(test_data, full_text)
        matcher_results = []
        for result in ner_results:
            entities = []
            for (ent_start, ent_stop, ent_label,
                 personal_title) in result['entities']:
                person = result['content'][ent_start:ent_stop]
                final_match = self.find_match_for_person(
                    person, personal_title, characters)
                if final_match is not None:
                    entities.append([ent_start, ent_stop, final_match])

            matcher_results.append(
                {'content': result["content"], 'entities': entities})

        return matcher_results

    def find_match_for_person(self, person, personal_title, characters):
        potential_matches = []
        for index, character in enumerate(characters):
            ratio_no_title = fuzz.ratio(person, character)
            ratio = fuzz.ratio(
                ((personal_title + " ") if personal_title is not None else "") + person,
                character)
            partial_ratio = get_partial_ratio_for_all_permutations(
                person, character)
            if ratio == 100 or ratio_no_title == 100:
                potential_matches = [[character, ratio]]
                break
            if partial_ratio >= self.partial_ratio_precision:
                potential_matches.append([character, partial_ratio])

        potential_matches = sorted(
            potential_matches,
            key=lambda x: x[1],
            reverse=True)

        final_match = self.choose_best_match(
            person, personal_title, potential_matches, characters)
        if final_match is None:
            return None

        return final_match

    def choose_best_match(self, person, personal_title,
                          potential_matches, characters):
        if len(potential_matches) > 1:
            final_match = self.handle_multiple_potential_matches(
                person, personal_title, potential_matches)
        elif len(potential_matches) == 1:
            final_match = potential_matches[0][0]
        else:
            final_match = "PERSON"
            potential_names_from_diminutive = get_names_from_diminutive(person)
            if potential_names_from_diminutive is not None:
                for character in characters:
                    for name in potential_names_from_diminutive:
                        if name in character.lower().split():
                            return character

        return final_match

    def handle_multiple_potential_matches(
            self, person, personal_title, potential_matches):
        final_match = None
        if personal_title is not None:
            if personal_title == "the":
                return "the " + person
            else:
                title_gender = self.titles_gender_dict[personal_title][0]
                for match in potential_matches:
                    if get_name_gender(match[0]) == title_gender:
                        final_match = match[0]
                        break
        else:
            final_match = potential_matches[0][0]

        return final_match


def get_partial_ratio_for_all_permutations(potential_match, character_name):
    character_name_components = character_name.split()
    character_name_permutations = list(
        itertools.permutations(character_name_components))
    partial_ratios = []
    for permutation in character_name_permutations:
        partial_ratios.append(
            fuzz.partial_ratio(
                ' '.join(permutation),
                potential_match))
    return max(partial_ratios)
