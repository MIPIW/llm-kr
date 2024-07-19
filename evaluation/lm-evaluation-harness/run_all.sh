#!/bin/bash
# export CUDA_VISIBLE_DEVICES=$2 # 어차피 하나로 돌릴 거임
export TOKENIZERS_PARALLELISM=false

RESULT_DIR='../../outputs/benchmarks/all'
TASKS='haerae_general_knowledge,haerae_history,haerae_loan_word,haerae_rare_word,haerae_standard_nomenclature'
MODEL=$1 #/home/beomi/coding-ssd2t/EasyLM/llama-2-ko-7b
OUTPUT_PATH=$(echo $MODEL | awk -F/ '{print $(NF-3) "/" $(NF-2) "/" $(NF-1)}')
SUB_FOLDER=$4

LORA="lora"
DEFAULT="default"

echo "mkdir -p $RESULT_DIR/${MODEL_PATH}/${SUB_FOLDER}/"
mkdir -p $RESULT_DIR/${MODEL_PATH}/${SUB_FOLDER}/


if [ "$2" = ${LORA} ]; then
    echo "lora initialized"

    python3 -m lm_eval \
        --model hf \
        --model_args pretrained=${3},peft=$MODEL \
        --tasks $TASKS \
        --device cuda:0 \
        --batch_size auto:4 \
        --output_path $RESULT_DIR/${MODEL_PATH}/${SUB_FOLDER}/haerae.json \

fi 

if [ "$2" = $DEFAULT ]; then 
    echo "default initialized"

    python3 -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL} \
        --tasks $TASKS \
        --device cuda:0 \
        --batch_size auto:4 \
        --output_path $RESULT_DIR/${MODEL_PATH}/${SUB_FOLDER}/haerae.json \

fi