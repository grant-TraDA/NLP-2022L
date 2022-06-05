import flair
from flair.models import SequenceTagger
from flair.data import Sentence
import torch

from tool.model.ner_model import NERModel


class FlairModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):

        super().__init__(save_personal_titles, fix_personal_titles)
        flair.device = torch.device('cpu')
        self.model = SequenceTagger.load(model_path)
        print('Flair model "' + model_path + '" loaded.')

    def get_doc_entities(self, text):

        doc = Sentence(text)
        self.model.predict(doc)

        entities = []
        for ent in doc.get_spans('ner'):
            if ent.labels[0].to_dict()['value'] == 'PER':
                ent_text = text[ent.start_position:ent.end_position]
                if self.fix_personal_titles and ent_text.startswith(
                        self.personal_titles):
                    ent.start_pos += (1 + len(ent_text.split(' ')[0]))
                if self.save_personal_titles:
                    personal_title = self.recognize_personal_title(ent, doc)
                    entities.append(
                        [ent.start_position, ent.end_position, "PERSON", personal_title])
                else:
                    entities.append(
                        [ent.start_position, ent.end_position, "PERSON"])

        return {'content': text, 'entities': entities}

    def recognize_personal_title(self, ent, doc):
        personal_title = None
        token_id = ent[0].idx - 1
        if token_id > 0:
            word_before_name = doc.tokens[token_id - 1].text
            if word_before_name.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.replace(".", "")
            if word_before_name.lower() == "the":
                personal_title = "the"
            return personal_title
