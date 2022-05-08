#!/bin/bash
set -e
cd ./../../

# argument1 - name of experiment; subdirectory of 'experiments' directory

for model in pl_core_news_sm pl_core_news_md pl_core_news_lg xx_ent_wiki_sm; do
  python -m tool.scripts.annotate_ner_pretrained data/novels_titles/polish_test_2.txt data/testing_sets/test experiments/"$1"/spacy__${model}/ experiments/tuned_ner_3/ ${model}
done