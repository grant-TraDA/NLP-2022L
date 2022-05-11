
import pickle

class DataLoader:

    def load(data_type, directory="data"):
        with open(f"{directory}/{data_type}_set_only_text.txt", "r") as file:
            x = file.readlines()
        with open(f"{directory}/{data_type}_set_only_tags.txt", "r") as file:
            y = list(map(int, file.readlines()))
        return x,y

    def save_bert_features(features, type, directory="data", bert="large"):
        with open(f"{directory}/{type}_x_bert_{bert}.pickle", "wb") as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_bert_features(type, directory="data", bert="large"):
        with open(f"./{directory}/{type}_x_bert_{bert}.pickle", "rb") as handle:
            return pickle.load(handle)
