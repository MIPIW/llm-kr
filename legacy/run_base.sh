#!/bin/bash

python3 train.py --train_batch_size 32 --eval_batch_size 32 --lora_type dora --lora_target no_qk --model gemma2-9b
python3 train.py --train_batch_size 32 --eval_batch_size 32 --lora_type dora --lora_target no_v --model gemma2-9b
python3 train.py --train_batch_size 32 --eval_batch_size 32 --lora_type dora --lora_target no_qk --model qwen2-7b
python3 train.py --train_batch_size 32 --eval_batch_size 32 --lora_type dora --lora_target no_v --model qwen2-7b
python3 train.py --train_batch_size 32 --eval_batch_size 32 --lora_type dora --lora_target no_qk --model llama3-8b
python3 train.py --train_batch_size 32 --eval_batch_size 32 --lora_type dora --lora_target no_v --model llama3-8b

