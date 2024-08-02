#!/bin/bash

# watchout for the results are not be merged
# echo gemma is running
# sh run_all.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/google/gemma-2-9b-it/default/checkpoint-9000' '0,1,2,3' 'default' 'NULL' '1epoch'
# sh run_all.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/google/gemma-2-9b-it/lora/checkpoint-4500' '0,1,2,3' 'lora' 'google/gemma-2-9b-it' '1epoch'

# echo qwen is running
# sh run_all.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/Qwen/Qwen2-7B-Instruct/default/checkpoint-4500' '0,1,2,3' 'default' 'NULL' '1epoch'
# sh run_all.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/Qwen/Qwen2-7B-Instruct/lora/checkpoint-4500' '0,1,2,3' 'lora' 'Qwen/Qwen2-7B-Instruct' '1epoch'

# echo llama is running
# sh run_all.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/default/checkpoint-4500' '0,1,2,3' 'default' 'NULL' '1epoch'
# sh run_all.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/lora/checkpoint-4500' '0,1,2,3' 'lora' 'meta-llama/Meta-Llama-3-8B-Instruct' '1epoch'


# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/qwen2-7b/dora/checkpoint-1065' '0,1,2,3' 'lora' 'Qwen/Qwen2-7B-Instruct' 
sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/llama3-8b/dora/checkpoint-1065' '0,1,2,3' 'lora' 'meta-llama/Meta-Llama-3-8B-Instruct'
sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/gemma2-9b/dora/checkpoint-1065' '0,1,2,3' 'lora' 'google/gemma-2-9b-it' 