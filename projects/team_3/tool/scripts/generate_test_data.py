import argparse

from tool.data_generator import generate_sample_test_data
from tool.file_and_directory_management import read_file_to_list
from tool.file_and_directory_management import dir_path, file_path


# titles_path - path to .txt file with titles of novels from which the sampled data are to be generated (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# novels_texts_dir_path - path to directory with .txt files containing full texts of novels (names of files should be
#       the same as titles on the list from titles_path)
# number_of_sentences - umber of sentences containing at least one named entity recognized as PERSON by
#       standard NER model to be randomly extracted from each novel
# generated_data_dir - directory where generated data should be stored
def main(titles_path, novels_texts_dir_path,
         number_of_sentences, generated_data_dir):
    titles = read_file_to_list(titles_path)

    generate_sample_test_data(
        titles,
        number_of_sentences,
        novels_texts_dir_path,
        generated_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path)
    parser.add_argument('novels_texts_dir_path', type=dir_path)
    parser.add_argument('number_of_sentences', type=int)
    parser.add_argument('generated_data_dir', type=str)
    opt = parser.parse_args()
    main(
        opt.titles_path,
        opt.novels_texts_dir_path,
        opt.number_of_sentences,
        opt.generated_data_dir)
