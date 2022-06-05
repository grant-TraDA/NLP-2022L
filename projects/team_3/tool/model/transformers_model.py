from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

import numpy as np

from tool.model.ner_model import NERModel


class TransformerModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):
        super().__init__(save_personal_titles, fix_personal_titles)
        if model_path == 'jplu/tf-xlm-r-ner-40-lang':
            self.model = pipeline("ner", model=model_path,
                                  tokenizer=(model_path, {"use_fast": True}),
                                  framework="tf",
                                  aggregation_strategy='simple')
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model = pipeline(
                "token-classification",
                aggregation_strategy="simple",
                model=model,
                tokenizer=tokenizer)
        print('Transformers model loaded.')

    def get_doc_entities(self, text):
        results = self.model(text)
        entities = []
        for index, ent in enumerate(results):
            if ent['entity_group'] == "PER":
                start, end = ent['start'], ent['end']
                if text[start] == ' ':
                    start += 1
                while end < len(text) and text[end].isalpha():
                    end += 1
                ent_text = text[start:end]
                if self.fix_personal_titles and ent_text.startswith(
                        self.personal_titles):
                    start += (1 + len(ent_text.split(' ')[0]))
                if self.save_personal_titles:
                    entities.append([start, end, "PERSON", None])
                else:
                    entities.append([start, end, "PERSON"])

        return {'content': text, 'entities': entities}
