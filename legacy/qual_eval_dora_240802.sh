#!/bin/bash

# 3개 instruction tuned 모델 대상 정성평가 파일 생성
# 실행 방법
# 1. 권한 설정: chmod +x qual_eval.sh
# 2. 실행: bash qual_eval.sh

# 평가 대상 모델
# /node_storage2/bigdata_IT_SFT/outputs/checkpoints/llama3-8b/dora/checkpoint-1065
# /node_storage2/bigdata_IT_SFT/outputs/checkpoints/qwen2-7b/dora/checkpoint-1065
# /node_storage2/bigdata_IT_SFT/outputs/checkpoints/gemma2-9b/dora/checkpoint-1065


### llama 3.1
python3 module_inference_revised_240711.py --model_path "../../../../node_storage2/bigdata_IT_SFT/outputs/checkpoints/llama3.1-8b/dora/checkpoint-1065/" --prompt_path "./prompt_240711.txt" --seed 1 --max_new_tokens 256 --temperature 0.8 --format_model meta-llama/Meta-Llama-3-8B-Instruct --peft

#############

model_dir="../../../../node_storage2/bigdata_IT_SFT/outputs/checkpoints/"
model_list="qwen2-7b/dora/checkpoint-1065 llama3-8b/dora/checkpoint-1065 gemma2-9b/dora/checkpoint-1065" # whitespace로 구분 
model_base="Qwen/Qwen2-7B-Instruct meta-llama/Meta-Llama-3-8B-Instruct google/gemma-2-9b-it"
model_strs=($model_base)
i=0

prompt_path="./prompt_240711.txt"

for model_path in $model_list
do
    echo ${model_strs[$i]}
    echo "--------- inference on" $model_dir$model_path "with" $prompt_path
    if [ $model_path = "meta-llama/Meta-Llama-3-8B-Instruct" ] ; then # config2: llama3-8b-it
        echo "config2 - llama3"
        python3 module_inference_revised_240711.py --model_path $model_dir$model_path --prompt_path $prompt_path --seed 1 --max_new_tokens 256 --temperature 0.6 --top_p 0.9 --top_k 50 --format_model ${model_strs[$i]} --peft
    
    else # config1 (default)
        echo "config1 - default"
        python3 module_inference_revised_240711.py --model_path $model_dir$model_path --prompt_path $prompt_path --seed 1 --max_new_tokens 256 --temperature 0.8 --format_model ${model_strs[$i]} --peft
    fi
    
    echo ""
    i=`expr $i + 1`
    
done
