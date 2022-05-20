def load_model(library, model_name):
    if library == 'coreferee':
        from tool.coreference.coreferee import Coreferee
        if model_name is None:
            model_name = 'pl_core_news_lg'
        model = Coreferee(model_name)

    else:
        raise Exception(
            'Library "' +
            library +
            '" is not supported. You can choose one of: coreferee')
    return model
