import argparse
import os
import json
from tqdm import tqdm

from tool.names_matcher import NamesMatcher
from tool.preprocessing import get_test_data_for_novel, get_characters_for_novel
from tool.file_and_directory_management import open_path, read_file_to_list, dir_path, file_path


def main(titles_path, characters_lists_dir_path, testing_data_dir_path, generated_data_dir,
         library, ner_model, fix_personal_titles, full_text, precision=75):
    titles = read_file_to_list(titles_path)
    names_matcher = NamesMatcher(
        precision,
        library,
        ner_model,
        fix_personal_titles)

    for title in tqdm(titles):
        test_data = get_test_data_for_novel(
            title, testing_data_dir_path, full_text)
        characters = get_characters_for_novel(title, characters_lists_dir_path)
        matcher_results = names_matcher.recognize_person_entities(
            test_data, characters, full_text)
        path = os.path.join(generated_data_dir, title + '.json')
        open_path(path, 'w').write(json.dumps(matcher_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('characters_lists_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with novel characters")
    parser.add_argument('testing_data_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with sentences extracted from novels "
                             "to be included in the testing process")
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('library', type=str, default='spacy', nargs='?',
                        help="library which should be used to test NER")
    parser.add_argument('ner_model', type=str, default=None, nargs='?',
                        help="model from chosen library which should be used to test NER")
    parser.add_argument('--fix_personal_titles', action='store_true')
    parser.add_argument('--full_text', action='store_true')
    opt = parser.parse_args()

    main(opt.titles_path, opt.characters_lists_dir_path, opt.testing_data_dir_path,
         opt.generated_data_dir, opt.library, opt.ner_model, opt.fix_personal_titles, opt.full_text)
