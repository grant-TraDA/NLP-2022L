from tool.names_matcher import NamesMatcher


def get_word_clusters(doc):
    word_clusters = []
    for idx_cluster in doc._.coref_chains:
        word_clusters.append(idx_cluster_to_word_cluster(idx_cluster, doc))

    return word_clusters


def idx_cluster_to_word_cluster(idx_cluster, doc):
    word_cluster = []
    for word_id in idx_cluster:
        word_id = word_id[0]
        word_cluster.append(doc[word_id])
    return word_cluster


class ClusterMatcher:

    def __init__(self, library, model_path):
        self.names_matcher = NamesMatcher(
            library='spacy',
            model_path=model_path,
            partial_ratio_precision=75)

    def match_single_cluster_with_character(self, cluster, characters):
        match = None
        for word in cluster:
            match = self.names_matcher.find_match_for_person(str(word),
                                                             personal_title=None,
                                                             characters=characters)
            if match is not None:
                break
        return match

    def match_clusters_with_characters(self, characters, clusters):
        matched_characters = []
        for cluster in clusters:
            match = self.match_single_cluster_with_character(
                cluster, characters)
            if match is None:
                matched_characters.append('PERSON')
            else:
                matched_characters.append(match)

        return matched_characters

    def annotated_coreference_json(self, doc, characters):
        text_dict = {'text': doc.text, 'id': 0, 'annotations': []}

        word_clusters = get_word_clusters(doc)
        matched_characters = self.match_clusters_with_characters(characters,
                                                                 word_clusters)

        for cluster_id, cluster in enumerate(doc._.coref_chains):
            for word_id in cluster:
                word_id = word_id[0]
                end = doc[word_id + 1].idx - 1 if not doc[word_id + 1].is_punct \
                    else doc[word_id + 1].idx
                label_dict = {'start': doc[word_id].idx,
                              'end': end,
                              'text': str(doc[word_id]),
                              'label': [matched_characters[cluster_id]][0]}
                text_dict['annotations'].append(label_dict)

        return text_dict
