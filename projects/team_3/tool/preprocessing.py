import os

from tool.file_and_directory_management import read_file, read_file_to_list, read_sentences_from_file


def get_litbank_text_parts(path):
    text = read_file(path)
    text_parts = text.split('\n\n\n')
    text_parts = ['\n\n'.join([' '.join(paragraph.split('\n')) for paragraph in part.split('\n\n')])
                  for part in text_parts]
    return text_parts


def get_litbank_text(path):
    text_parts = get_litbank_text_parts(path)
    text = '\n\n\n'.join(text_parts)
    return text


def get_test_data_for_novel(title, testing_data_dir_path, full_text):
    if full_text:
        return get_litbank_text(os.path.join(testing_data_dir_path, title))
    else:
        return read_sentences_from_file(
            os.path.join(testing_data_dir_path, title))


def get_characters_for_novel(title, characters_lists_dir_path):
    return read_file_to_list(os.path.join(characters_lists_dir_path, title))
