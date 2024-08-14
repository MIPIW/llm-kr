#!/bin/bash

# 3개 instruction tuned 모델 대상 정성평가 파일 생성
# 실행 방법
# 1. 권한 설정: chmod +x qual_eval.sh
# 2. 실행: bash qual_eval.sh

# model_list="Qwen/Qwen2-7B-Instruct meta-llama/Meta-Llama-3-8B-Instruct google/gemma-2-9b-it"
# i=0

prompt_path="./prompt_240711.txt"

python3 module_inference_revised_240711.py --model_path "meta-llama/Meta-Llama-3.1-8B-Instruct" --prompt_path $prompt_path --seed 1 --max_new_tokens 256 --temperature 0.8 --format_model "meta-llama/Meta-Llama-3.1-8B-Instruct"

python3 module_inference_revised_240711.py --model_path "../../../../node_storage2/bigdata_IT_SFT/outputs/checkpoints/llama3.1-8b/dora/checkpoint-1065/" --prompt_path "./prompt_240711.txt" --seed 1 --max_new_tokens 256 --temperature 0.8 --format_model "meta-llama/Meta-Llama-3.1-8B-Instruct" --peft



# for model_path in $model_list
# do
#     echo "--------- inference on" $model_path "with" $prompt_path
#     if [ $model_path = "meta-llama/Meta-Llama-3-8B-Instruct" ] ; then # config2: llama3-8b-it
#         echo "config2 - llama3"
#         python3 module_inference_revised_240711.py --model_path $model_path --prompt_path $prompt_path --seed 1 --max_new_tokens 256 --temperature 0.6 --top_p 0.9 --top_k 50 --format_model $model_path
        
#     else # config1 (default)
#         echo "config1 - default"
#         python3 module_inference_revised_240711.py --model_path $model_path --prompt_path $prompt_path --seed 1 --max_new_tokens 256 --temperature 0.8 --format_model $model_path
#     fi
    
#     echo ""
#     i=`expr $i + 1`
    
# done

