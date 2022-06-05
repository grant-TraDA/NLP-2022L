import torch
from autoencoders import Autoencoder
from nltk.corpus import stopwords
from flair.embeddings import WordEmbeddings
from nltk.tokenize import RegexpTokenizer
import nltk
from flask import Flask, jsonify, request
from flair.data import Sentence

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
max_len = 32
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\S+')
embedding = WordEmbeddings('glove')

app = Flask(__name__)

vae, _ = torch.load(
    'models/vae_big_18epoch.pt',
    map_location=torch.device('cpu'))

m = Autoencoder(100, 100, 100, 4, 3, variational=True, max_log2len=8)
m.load_state_dict(vae)
m = m.encoder.eval()


def __preprocess_sentenc(sentence):
    sentence = " ".join([
        word
        for word, tag in nltk.pos_tag(tokenizer.tokenize(sentence))
        if word not in stop_words
    ])
    sentence = Sentence(sentence)
    embedding.embed(sentence)
    embedding_size = sentence[0].embedding.shape[0]
    return torch.cat([
        torch.stack([t.embedding for t in sentence]),
        torch.zeros(max_len - len(sentence), embedding_size, device='cpu')
    ])


@app.route('/predict', methods=['POST'])
def hello_world():
    sentence = request.get_json()['sentence']
    if not isinstance(sentence, str) or len(sentence.split(' ')) > max_len:
        return "Bad request", 400

    with torch.no_grad():
        t = m.forward(__preprocess_sentenc(sentence)[None, :])
        return jsonify(t.detach().numpy().tolist()), 200
