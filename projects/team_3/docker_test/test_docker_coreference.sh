#!/bin/bash

## ANNOTATE COREFERENCE
set -e

python -m tool.scripts.annotate_coreference data/novels_titles/coreference_titles.txt data/lists_of_characters data/testing_sets/test_coreference data/results/coreference coreferee pl_core_news_lg