#!/bin/bash
set -e

# argument1 - name of experiment; subdirectory of 'experiments' directory
# argument2 - flag; if '--fixing_titles' then personal titles annotations will be fixed

for model in pl_core_news_sm pl_core_news_md pl_core_news_lg xx_ent_wiki_sm; do
  python -m tool.scripts.annotate_protagonist data/novels_titles/polish_titles.txt data/lists_of_characters data/testing_sets/test experiments/"$1"/spacy__${model}/ spacy ${model} "${@:2}"
done

for model in ner-multi ner-multi-fast; do
  python -m tool.scripts.annotate_protagonist data/novels_titles/polish_titles.txt data/lists_of_characters data/testing_sets/test experiments/"$1"/flair__${model}/ flair ${model} "${@:2}"
done

for model in jplu/tf-xlm-r-ner-40-lang Davlan/distilbert-base-multilingual-cased-ner-hrl Davlan/bert-base-multilingual-cased-ner-hrl; do
  python -m tool.scripts.annotate_protagonist data/novels_titles/polish_titles.txt data/lists_of_characters data/testing_sets/test experiments/"$1"/transformers__${model/*\//}/ transformers ${model} "${@:2}"
done