import argparse

from tool.ner_metrics import metrics
from tool.file_and_directory_management import dir_path, file_path


def main(titles_path, gold_standard_dir_path,
         testing_set_dir_path, stats_dir, protagonist_tagger=False, print_results=False, debug_mode=False):
    metrics(
        titles_path,
        gold_standard_dir_path,
        testing_set_dir_path,
        stats_dir,
        protagonist_tagger,
        print_results,
        debug_mode
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('gold_standard_dir_path', type=dir_path,
                        help="path to the directory with .json files containing goldstandard entities")
    parser.add_argument('testing_set_dir_path', type=dir_path,
                        help="path to the directory with .json files containing predicted entities")
    parser.add_argument('stats_dir', type=str,
                        help="directory where the computed statistics should be stored")
    parser.add_argument('--protagonist_tagger', action='store_true',
                        help="if metrics for protagonist_tagger should be calculated")
    parser.add_argument('--print_results', action='store_true',
                        help="if calculated results should be printed to the console")
    parser.add_argument('--debug_mode', action='store_true')
    opt = parser.parse_args()
    main(opt.titles_path, opt.gold_standard_dir_path, opt.testing_set_dir_path,
         opt.stats_dir, opt.protagonist_tagger, opt.print_results, opt.debug_mode)
