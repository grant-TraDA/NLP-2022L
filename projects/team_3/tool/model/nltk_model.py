import nltk

from tool.model.ner_model import NERModel


class NLTKModel(NERModel):

    def __init__(self, save_personal_titles, fix_personal_titles):

        super().__init__(save_personal_titles, fix_personal_titles)
        print('NLTK model loaded.')

    def get_doc_entities(self, text):

        entities = []
        offset = 0
        spans = []

        for token in nltk.word_tokenize(text):
            offset = text.find(token, offset)
            spans.append((offset, offset + len(token)))

        for chunk_id, chunk in enumerate(nltk.ne_chunk(
                nltk.pos_tag(nltk.word_tokenize(text)))):

            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                try:
                    start, end = spans[chunk_id]
                    ent_text = text[start:end]
                    if self.fix_personal_titles and ent_text.startswith(
                            self.personal_titles):
                        start += (1 + len(ent_text.split(' ')[0]))
                    if self.save_personal_titles:
                        personal_title = self.recognize_personal_title(
                            text, chunk_id)
                        entities.append([start, end, "PERSON", personal_title])
                    else:
                        entities.append([start, end, "PERSON"])
                except IndexError:
                    pass

        return {'content': text, 'entities': entities}

    def recognize_personal_title(self, text, chunk_id):
        personal_title = None
        if chunk_id > 0:
            word_before_name = nltk.word_tokenize(text)[chunk_id - 1]
            if word_before_name.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.replace(".", "")
            if word_before_name.lower() == "the":
                personal_title = "the"
        return personal_title
