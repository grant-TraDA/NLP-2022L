import argparse
import os
from tqdm import tqdm

from tool.file_and_directory_management import read_file_to_list, dir_path, \
    file_path, write_json
from tool.coreference.utils import load_model
from tool.preprocessing import get_test_data_for_novel, \
    get_characters_for_novel
from tool.coreference_cluster_utils import ClusterMatcher


def main(titles_path, characters_lists_dir_path,
         testing_data_dir_path, generated_data_dir,
         library, model_name):
    titles = read_file_to_list(titles_path)
    model = load_model(library, model_name)
    cluster_matcher = ClusterMatcher(library, model_name)

    for title in tqdm(titles):
        characters = get_characters_for_novel(title, characters_lists_dir_path)
        test_data = get_test_data_for_novel(title, testing_data_dir_path,
                                            False)
        fragments = []
        for fragment in test_data:
            doc = model.get_doc(fragment)

            fragment_json = cluster_matcher.annotated_coreference_json(doc,
                                                                       characters)
            fragments.append(fragment_json)
        path = os.path.join(generated_data_dir, title + '.json')

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        write_json(fragments, path)


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
    parser.add_argument('library', type=str, default='coreferee', nargs='?',
                        help="library which should be used for coreferences")
    parser.add_argument('model_name', type=str, default=None, nargs='?',
                        help="model from chosen library which should be used")
    opt = parser.parse_args()
    main(opt.titles_path, opt.characters_lists_dir_path,
         opt.testing_data_dir_path, opt.generated_data_dir,
         opt.library, opt.model_name)
