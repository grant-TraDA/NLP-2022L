
import re
from tqdm import tqdm

from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils

RE_EMOJI = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)

class Roberta:
    """
    Initializes RoBERTa model.
    Arguments:
        - path: a string (default: "pretrained-models/roberta/large")
    Notes:
        If the model is not found at the given path, the function throws an error.
    """
    def __init__(self, path="pretrained-models/roberta/large"):
        loaded = hub_utils.from_pretrained(
            model_name_or_path=path,
            data_name_or_path=path,
            bpe="sentencepiece",
            sentencepiece_vocab="sentencepiece.bpe.model",
            load_checkpoint_heads=True,
            archive_map=RobertaModel.hub_models(),
            cpu=True
        )
        self.roberta = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
        self.roberta.eval()

    def preprocess_text(self, input):
        text = input.strip()
        if text.startswith("RT"):
            text = text[len("RT"):].lstrip()
        while text.startswith("@anonymized_account"):
            text = text[len("@anonymized_account"):].lstrip()
        while text.endswith("@anonymized_account"):
            text = text[:-len("@anonymized_account")].rstrip()
        text = RE_EMOJI.sub(r"", text)
        return text

    def extract_features(self, x, verbose=True):
        vectors = []
        for xi in tqdm(x) if verbose else x:
            tokens = self.roberta.encode(xi)
            features = self.roberta.extract_features(tokens, return_all_hiddens=True)[-2].cpu().detach().numpy().squeeze()
            vectors.append(features.mean(axis=0))
        return vectors

    def preprocess_extract_features(self, x):
        x_processed = [self.preprocess_text(xi) for xi in x]
        return self.extract_features(x_processed)
