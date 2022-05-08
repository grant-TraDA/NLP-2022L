from tqdm import tqdm


class NERModel:
    def __init__(self, save_personal_titles, fix_personal_titles):
        self.save_personal_titles = save_personal_titles
        self.fix_personal_titles = fix_personal_titles
        if save_personal_titles or fix_personal_titles:
            from tool.gender_checker import get_personal_titles
            self.personal_titles = tuple(get_personal_titles())

    def get_ner_results(self, data, full_text=False):
        if full_text:
            results = [self.get_doc_entities(data)]
        else:
            results = []
            for sentence in tqdm(data, leave=False):
                results.append(self.get_doc_entities(sentence))
        return results

    def get_doc_entities(self, text):
        pass
