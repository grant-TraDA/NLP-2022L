from poldeepner2 import models

from tool.model.ner_model import NERModel


class PolDeepNerModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):
        super().__init__(save_personal_titles, fix_personal_titles)
        self.model = models.load(
            model_path,
            device="cpu",
            resources_path="poldeepner_resources")
        print('PolDeepNer model "' + model_path + '" loaded.')

    def get_doc_entities(self, text):
        doc = self.model.process_text(text)
        entities = []
        for index, ent in enumerate(doc):
            # if ent.label in ["nam_liv_animal", "nam_liv_character", "nam_liv_god", "nam_liv_habitant"]:
            #     print(ent)
            if ent.label == "persName" or ent.label == 'nam_liv_person':
                start, end = ent.begin, ent.end
                ent_text = text[start:end]
                if self.fix_personal_titles and ent_text.startswith(
                        self.personal_titles):
                    start += (1 + len(ent_text.split(' ')[0]))
                if self.save_personal_titles:
                    personal_title = self.recognize_personal_title(doc, index)
                    entities.append([start, end, "PERSON", personal_title])
                else:
                    entities.append([start, end, "PERSON"])

        return {'content': text, 'entities': entities}

    def recognize_personal_title(self, doc, index):
        pass
