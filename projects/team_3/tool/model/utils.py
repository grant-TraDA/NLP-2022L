def load_model(library, ner_model, save_personal_titles,
               fix_personal_titles=True):
    if library == 'spacy':
        from tool.model.spacy_model import SpacyModel
        if ner_model is None:
            ner_model = 'en_core_web_sm'
        model = SpacyModel(
            ner_model,
            save_personal_titles,
            fix_personal_titles)

    elif library == 'nltk':
        import nltk
        from tool.model.nltk_model import NLTKModel
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        model = NLTKModel(save_personal_titles, fix_personal_titles)

    elif library == 'stanza':
        from tool.model.stanza_model import StanzaModel
        model = StanzaModel(save_personal_titles, fix_personal_titles)

    elif library == 'flair':
        from tool.model.flair_model import FlairModel
        if ner_model is None:
            ner_model = 'ner'
        model = FlairModel(
            ner_model,
            save_personal_titles,
            fix_personal_titles)

    elif library == 'transformers':
        from tool.model.transformers_model import TransformerModel
        if ner_model is None:
            ner_model = "xlm-roberta-large-finetuned-conll03-english"
        model = TransformerModel(
            ner_model,
            save_personal_titles,
            fix_personal_titles)

    else:
        raise Exception('Library "' + library + '" is not supported. You can choose one of: spacy, nltk, stanza and '
                                                'flair.')
    return model
