import argparse
import os
import json
from tqdm import tqdm

from tool.file_and_directory_management import read_file_to_list, file_path


def main(titles_path, ner_data_dir, coreference_data_dir, generated_data_dir):
    titles = read_file_to_list(titles_path)

    for title in tqdm(titles):

        new_results = []

        with open(os.path.join(ner_data_dir, title + '.json')) as f:
            ner_results = json.loads(f.read())

        with open(os.path.join(coreference_data_dir, title + '.json')) as f:
            coref_results = json.loads(f.read())

        for cluster in coref_results['mentions']:
            possible_matches = []
            matches = []
            for ent in ner_results[0]['entities']:
                if ent[:2] in cluster:
                    matches.append(ent)
                    possible_matches.append(ent[2])
            if len(possible_matches) > 1:
                match = max(set(possible_matches), key=possible_matches.count)
                for mention in cluster:
                    new_results.append((mention[0], mention[1], match))
            if matches:
                print([ner_results[0]['content'][c[0]:c[1]] for c in cluster])
                for m in matches:
                    print(m, ner_results[0]['content'][m[0]:m[1]])

        results_dict = {
            'content': ner_results[0]['content'],
            'mentions': new_results}
        path = os.path.join(generated_data_dir, title + ".json")

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'w+') as result:
            json.dump(results_dict, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('ner_data_dir', type=str)
    parser.add_argument('coreference_data_dir', type=str)
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    opt = parser.parse_args()
    main(
        opt.titles_path,
        opt.ner_data_dir,
        opt.coreference_data_dir,
        opt.generated_data_dir)
