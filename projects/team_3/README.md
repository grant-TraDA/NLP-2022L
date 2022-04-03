The main contribution in this part of the project is preparation of data which are available in the `data` directory. Initial pre-annotations were obtained by running the script:

``
python -m tool.scripts.annotate_protagonist data/novels_titles/polish_titles.txt  data/lists_of_characters data/testing_sets/test experiments/polish spacy pl_core_news_lg
``

Converting the results to the form required by LabelStudio and vice versa, as well as the scripts for data analysis, are in the notebook in the `notebooks` folder.
