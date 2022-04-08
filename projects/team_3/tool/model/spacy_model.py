import spacy

from tool.model.ner_model import NERModel


class SpacyModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):
        super().__init__(save_personal_titles, fix_personal_titles)
        self.model = spacy.load(model_path)
        print('Spacy model "' + model_path + '" loaded.')

    def get_doc_entities(self, text):
        doc = self.model(text)
        entities = []
        for index, ent in enumerate(doc.ents):
            if ent.label_ == "persName":
                start, end = ent.start_char, ent.end_char
                ent_text = text[start:end]
                if self.fix_personal_titles and ent_text.startswith(
                        self.personal_titles):
                    start += (1 + len(ent_text.split(' ')[0]))
                if self.save_personal_titles:
                    personal_title = self.recognize_personal_title(doc, index)
                    entities.append([start, end, "persName", personal_title])
                else:
                    entities.append([start, end, "persName"])

        return {'content': text, 'entities': entities}

    def recognize_personal_title(self, doc, index):
        personal_title = None
        span = doc.ents[index]
        if span.start > 0:
            word_before_name = doc[span.start - 1].text
            if word_before_name.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.replace(".", "")
            if word_before_name.lower() == "the":
                personal_title = "the"

        return personal_title
