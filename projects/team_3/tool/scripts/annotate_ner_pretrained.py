import argparse
import os
import json
import spacy
from tqdm import tqdm

from tool.file_and_directory_management import open_path, read_file_to_list, \
    dir_path, file_path
from tool.model.utils import load_model
from tool.preprocessing import get_test_data_for_novel


def main(titles_path, testing_data_dir_path, generated_data_dir, model_output_dir='experiments/tuned_ner/',
         model_name='pl_core_news_lg', fix_personal_titles=False, full_text=False):
    titles = read_file_to_list(titles_path)
    model_path = os.path.join(model_output_dir, model_name)
    model = spacy.load(model_path)

    for title in tqdm(titles):
        test_data = get_test_data_for_novel(
            title, testing_data_dir_path, full_text)
        ner_result = []

        for text in test_data:
            entities = []
            doc = model(text)
            tmp_text = text

            for ent in doc.ents:
                start = tmp_text.find(ent.text)
                end = start + len(ent.text)
                entities.append([start, end, ent.label_])
                tmp_text = tmp_text[end:]

            ner_result.append({"content": text, "entities": entities})

        path = os.path.join(generated_data_dir, title + ".json")
        open_path(path, 'w').write(json.dumps(ner_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('testing_data_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with sentences extracted from novels "
                             "to be included in the testing process")
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('model_output_dir', type=str, default='experiments/tuned_ner/',
                        help="path to the directory containing fine tuned NER models' folders")
    parser.add_argument('model_name', type=str, default='pl_core_news_lg',
                        help="name of folder containing the fine tuned model which should be used to test NER")
    parser.add_argument('--full_text', action='store_true')
    opt = parser.parse_args()
    main(opt.titles_path, opt.testing_data_dir_path,
         opt.generated_data_dir, opt.model_output_dir, opt.model_name, opt.full_text)
