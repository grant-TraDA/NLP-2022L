#!/bin/bash
set -e

# ANNOTATE NER
for model in pl_core_news_sm pl_core_news_md pl_core_news_lg xx_ent_wiki_sm; do
  python -m tool.scripts.annotate_ner docker_test/titles.txt docker_test/ docker_test/experiments/ner/spacy__${model}/ spacy ${model}
done

for model in ner-multi ner-multi-fast; do
  python -m tool.scripts.annotate_ner docker_test/titles.txt docker_test docker_test/experiments/ner/flair__${model}/ flair ${model}
done

for model in jplu/tf-xlm-r-ner-40-lang Davlan/distilbert-base-multilingual-cased-ner-hrl Davlan/bert-base-multilingual-cased-ner-hrl; do
  python -m tool.scripts.annotate_ner docker_test/titles.txt docker_test docker_test/experiments/ner/transformers__${model/*\//}/ transformers ${model}
done

for model in kpwr-n82-base cen-n82-base cen-n82-large nkjp-base nkjp-base-sq conll-english-large-sq; do
  python -m tool.scripts.annotate_ner docker_test/titles.txt docker_test docker_test/experiments/ner/pdn2__${model/*\//}/ poldeepner ${model} "${@:2}"
done


# EVALUATE NER
for model in pl_core_news_sm pl_core_news_md pl_core_news_lg xx_ent_wiki_sm; do
  time python -m tool.scripts.compute_metrics docker_test/titles.txt docker_test/person_goldstandard docker_test/experiments/ner/spacy__${model}/ docker_test/experiments/ner/spacy__${model}/stats --print_results
done

for model in ner-multi ner-multi-fast; do
  time python -m tool.scripts.compute_metrics docker_test/titles.txt docker_test/person_goldstandard docker_test/experiments/ner/flair__${model}/ docker_test/experiments/ner/flair__${model}/stats --print_results
done

for model in jplu/tf-xlm-r-ner-40-lang Davlan/distilbert-base-multilingual-cased-ner-hrl Davlan/bert-base-multilingual-cased-ner-hrl; do
  time python -m tool.scripts.compute_metrics docker_test/titles.txt docker_test/person_goldstandard docker_test/experiments/ner/transformers__${model/*\//}/ docker_test/experiments/ner/transformers__${model/*\//}/stats --print_results
done

for model in kpwr-n82-base cen-n82-base cen-n82-large nkjp-base nkjp-base-sq conll-english-large-sq; do
  time python -m tool.scripts.compute_metrics docker_test/titles.txt docker_test/person_goldstandard docker_test/experiments/ner/pdn2__${model/*\//}/ docker_test/experiments/ner/pdn2__${model/*\//}/stats --print_results
done


# ANNOTATE PROTAGONIST
for model in pl_core_news_sm pl_core_news_md pl_core_news_lg xx_ent_wiki_sm; do
  python -m tool.scripts.annotate_protagonist docker_test/titles.txt docker_test/characters docker_test docker_test/experiments/protagonist/spacy__${model}/ spacy ${model}
done

for model in ner-multi ner-multi-fast; do
  python -m tool.scripts.annotate_protagonist docker_test/titles.txt docker_test/characters docker_test docker_test/experiments/protagonist/flair__${model}/ flair ${model}
done

for model in jplu/tf-xlm-r-ner-40-lang Davlan/distilbert-base-multilingual-cased-ner-hrl Davlan/bert-base-multilingual-cased-ner-hrl; do
  python -m tool.scripts.annotate_protagonist docker_test/titles.txt docker_test/characters docker_test docker_test/experiments/protagonist/transformers__${model/*\//}/ transformers ${model}
done

for model in kpwr-n82-base cen-n82-base cen-n82-large nkjp-base nkjp-base-sq conll-english-large-sq; do
  python -m tool.scripts.annotate_protagonist docker_test/titles.txt docker_test/characters docker_test docker_test/experiments/protagonist/pdn2__${model/*\//}/ poldeepner ${model}
done

# EVALUATE PROTAGONIST
for model in pl_core_news_sm pl_core_news_md pl_core_news_lg xx_ent_wiki_sm; do
  time python -m tool.scripts.compute_metrics docker_test/titles.txt docker_test/names_goldstandard docker_test/experiments/protagonist/spacy__${model}/ docker_test/experiments/protagonist/spacy__${model}/stats --print_results --protagonist_tagger
done

for model in ner-multi ner-multi-fast; do
  time python -m tool.scripts.compute_metrics docker_test/titles.txt docker_test/names_goldstandard docker_test/experiments/protagonist/flair__${model}/ docker_test/experiments/protagonist/flair__${model}/stats --print_results --protagonist_tagger
done

for model in jplu/tf-xlm-r-ner-40-lang Davlan/distilbert-base-multilingual-cased-ner-hrl Davlan/bert-base-multilingual-cased-ner-hrl; do
  time python -m tool.scripts.compute_metrics docker_test/titles.txt docker_test/names_goldstandard docker_test/experiments/protagonist/transformers__${model/*\//}/ docker_test/experiments/protagonist/transformers__${model/*\//}/stats --print_results --protagonist_tagger
done

for model in kpwr-n82-base cen-n82-base cen-n82-large nkjp-base nkjp-base-sq conll-english-large-sq; do
  time python -m tool.scripts.compute_metrics docker_test/titles.txt docker_test/names_goldstandard docker_test/experiments/protagonist/pdn2__${model/*\//}/ docker_test/experiments/protagonist/pdn2__${model/*\//}/stats --print_results --protagonist_tagger
done

