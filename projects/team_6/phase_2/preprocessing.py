from unidecode import unidecode
from stemming import stemming


def data_stemming(data):
    new_data = []
    for d in data:
        new_d = stemming(d)
        new_data.append(new_d)
    return new_data


def data_preprocessing(data):
    new_data = []
    for d in data:
        nd = d.lower()
        nd = unidecode(nd)
        nd = nd.replace('@anonymized_account', '')
        nd = nd.replace(',', '')
        nd = nd.replace('.', '')
        new_data.append(nd)
    return new_data


def remove_stopwords(data, stopwords):
    new_list = []
    for d in data:
        words = d.split()
        new_words = []
        for w in words:
            if w not in stopwords:
                new_words.append(w)
        new_d = ' '.join(new_words)
        new_list.append(new_d)
    return new_list


def full_data_preprocessing(data, stopwords, stem=False):
    if stem:
        data = data_stemming(data)
    data = data_preprocessing(data)
    data = remove_stopwords(data, stopwords)
    return data
