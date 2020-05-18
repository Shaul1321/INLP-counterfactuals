#!/usr/bin/env bash

CUDA_ID=0

for DATASET in bios; do # sst-2 sst-3 sst-5 lgd agnews news20 mtl_16 aclImdb2
    MODEL_PATH="/home/nlp/ravfogs/INLP-counterfactuals/src/experiments/models/bios/counterfactuals"
    OUTPUT_PATH=${MODEL_PATH}
#    OUTPUT_PATH="experiments/models"

    for SPLIT in train dev test; do
        DATA_PATH="/home/nlp/ravfogs/INLP-counterfactuals/data/bios/${SPLIT}.jsonl"

        CUDA_VISIBLE_DEVICES=${CUDA_ID} \
        allennlp predict ${MODEL_PATH}/model.tar.gz \
         ${DATA_PATH} \
         --overrides "{model: {output_hidden_states: true}}" \
         --predictor=jsonl_predictor \
         --cuda-device 0 \
         --include-package=inlpt > ${OUTPUT_PATH}/pred_${SPLIT}.txt
#         --output-file=${OUTPUT_PATH}/pred_${SPLIT}.txt \

        python experiments/scripts/save_cls.py -i=${OUTPUT_PATH}/pred_${SPLIT}.txt -o=${OUTPUT_PATH}
    done
done