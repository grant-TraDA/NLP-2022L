import pathlib
import torch

# max length of line
MAX_LENGTH = 20

MODEL_LR = 0.3
GPT2_LR = 0.001

ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent
DATA_PATH = ROOT_PATH / "verses_pairs.csv"
LOG_PATH = ROOT_PATH / 'logs'
LOG_PATH.mkdir(exist_ok=True)

# token ids in AccentTokenizer
PAD_token = 0
SOS_token = 1
EOS_token = 2

# device type
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
