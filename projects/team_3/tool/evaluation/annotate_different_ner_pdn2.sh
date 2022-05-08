#!/bin/bash
set -e

# argument1 - name of experiment; subdirectory of 'experiments' directory
# argument2 - flag; if '--fixing_titles' then personal titles annotations will be fixed

for model in kpwr-n82-base cen-n82-base cen-n82-large nkjp-base nkjp-base-sq conll-english-large-sq; do
  python -m tool.scripts.annotate_ner data/novels_titles/polish_titles.txt data/testing_sets/test experiments/"$1"/pdn2__${model/*\//}/ poldeepner ${model} "${@:2}"
done
