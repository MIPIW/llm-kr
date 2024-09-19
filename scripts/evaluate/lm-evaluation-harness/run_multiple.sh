#!/bin/bash

# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/google/gemma-2-9b-it/default/checkpoint-9000' 'default' 'X' 'base'
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/google/gemma-2-9b-it/lora/checkpoint-4500' 'lora' 'google/gemma-2-9b-it' 
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/Qwen/Qwen2-7B-Instruct/default/checkpoint-4500' 'default' 'X' 'base'
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/Qwen/Qwen2-7B-Instruct/lora/checkpoint-4500' 'lora' 'Qwen/Qwen2-7B-Instruct' 
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/default/checkpoint-4500' 'default' 'X' 'base'
# sh run.sh '/home/hyohyeongjang/bigdata/ModelTrainer/llm-kr/outputs/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/lora/checkpoint-4500' 'lora' 'meta-llama/Meta-Llama-3-8B-Instruct' 

# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/qwen2-7b/dora/checkpoint-1065' 'lora' 'Qwen/Qwen2-7B-Instruct'
# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/llama3-8b/dora/checkpoint-1065' 'lora' 'meta-llama/Meta-Llama-3-8B-Instruct'
# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/gemma2-9b/dora/checkpoint-1065' 'lora' 'google/gemma-2-9b-it' 

# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/llama3.1-8b/dora/checkpoint-1065/' 'lora' 'meta-llama/Meta-Llama-3.1-8B-Instruct' 
# sh run_all.sh 'meta-llama/Meta-Llama-3.1-8B-Instruct' 'default' 'X' 'base'

# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/gemma2-9b/dora/2of2/checkpoint-116' 'lora' 'google/gemma-2-9b-it'
# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/llama3.1-8b/dora/2of2/checkpoint-116' 'lora' 'meta-llama/Meta-Llama-3.1-8B-Instruct'
# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/llama3-8b/dora/2of2/checkpoint-116' 'lora' 'meta-llama/Meta-Llama-3-8B-Instruct'

# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/llama3.1-8b/no/2of2/checkpoint-116' 'default' 'X' '2of2'
# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/llama3-8b/no/2of2/checkpoint-116' 'default' 'X' '2of2'
# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/qwen2-7b/dora/2of2/checkpoint-116' 'lora' 'Qwen/Qwen2-7B-Instruct' 

# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/gemma2-9b/no/2of2/checkpoint-116' 'default' 'NULL' '2of2'
# sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/qwen2-7b/lora/2of2/checkpoint-116' 'lora' 'Qwen/Qwen2-7B-Instruct' '2of2' 

sh run_all.sh '/node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch/qwen2-7b/no/2of2/checkpoint-116' 'default' 'NULL' '2of2'