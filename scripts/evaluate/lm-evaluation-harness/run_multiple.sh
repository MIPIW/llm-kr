#!/bin/bash

# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/google/gemma-2-9b-it/default/checkpoint-9000' 'default' # '0,1,2,3'
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/google/gemma-2-9b-it/lora/checkpoint-4500' 'lora' 'google/gemma-2-9b-it' # '0,1,2,3'
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/Qwen/Qwen2-7B-Instruct/default/checkpoint-4500' 'default' # '0,1,2,3'
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/Qwen/Qwen2-7B-Instruct/lora/checkpoint-4500' 'lora' 'Qwen/Qwen2-7B-Instruct' # '0,1,2,3' 안돌아감
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/default/checkpoint-4500' 'default' # '0,1,2,3' not completed
sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/lora/checkpoint-4500' 'lora' 'meta-llama/Meta-Llama-3-8B-Instruct' # '0,1,2,3' 안돌아감 