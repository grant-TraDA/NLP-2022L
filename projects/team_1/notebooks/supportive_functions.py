import re
import pandas as pd


def find_nonalpha(text):
    result = re.findall(
        "[^AaĄąBbCcĆćDdEeĘęFfGgHhIiJjKkLlŁłMmNnŃńOoÓóPpRrSsŚśTtUuVvWwYyZzŹźŻżQqXx]",
        text
        )
    return result


def regex_filter(val, regex):
    if val:
        mo = re.search(regex, val)
        if mo:
            return True
        else:
            return False
    else:
        return False


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def count_punctuation(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Count number of occurences of each of the special characters
    in the data.

    Args:
        data: Dataframe to investigate, with second column
            containing text data.
        col: Name of the column in the data set contating text data.

    Returns:
        pd.DataFrame: Dataframe containing the counts of occurences
            of each special character in each record.
    """
    data_stats = data.copy()
    for index, row in data_stats.iterrows():
        data_stats.loc[index, 'fullstop'] = row[col].count('.')
        data_stats.loc[index, 'comma'] = row[col].count(',')
        data_stats.loc[index, 'question_mark'] = row[col].count('?')
        data_stats.loc[index, 'exclamation_mark'] = row[col].count('!')
        data_stats.loc[index, 'hyphen'] = row[col].count('-')
        data_stats.loc[index, 'colon'] = row[col].count(':')
        data_stats.loc[index, 'ellipsis'] = row[col].count('...')
        # data_stats.loc[index, 'semicolon'] = row[col].count(';')
        # data_stats.loc[index, 'quote'] = row[col].count('"')
    return data_stats.iloc[:, 1:]


def separate_special_chars(data: pd.DataFrame) -> pd.DataFrame:
    """
    Insert spaces before the special characters to separate them
    from the actual words.

    Args:
        data: Dataframe to process, with text data
            in the second column.

    Returns:
        pd.DataFrame: Input dataframe with special characters
            separated from the actual words.
    """
    data_sep = data.copy()
    col = data_sep.columns[1]
    special_chars = ['.', ',', '?', '!', '-', ':', ';', ' ']
    for index, row in data_sep.iterrows():
        for i in special_chars:
            if i in row[col]:
                row[col] = row[col].replace(i, ' '+i)
    return data_sep


def rm_consecutive_spaces(string):
    return re.sub(' {2,}', ' ', string)
