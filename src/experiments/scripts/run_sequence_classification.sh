#!/usr/bin/env bash

for DATASET in bios; do # sst-2 sst-3 sst-5 lgd agnews news20 mtl_16 aclImdb2
    MODEL_PATH="experiments/models/${DATASET}/counterfactual"

    allennlp train experiments/configs/${DATASET}.jsonnet \
     -s ${MODEL_PATH} -f \
     --include-package=model_files
    # use this flag to change number of nullspace classifiers
    # --overrides "{ model: { num_classifiers: 10 } }"

    #python experiments/scripts/save_w.py --model-path=${MODEL_PATH}

    find ${MODEL_PATH} -name "*.th" -delete
done