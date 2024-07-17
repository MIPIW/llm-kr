#!/bin/bash

echo google/gemma-2-9b-it running
python3 trainig_default.py \
    --model google/gemma-2-9b-it \
    --train_batch_size 16 \
    --eval_batch_size 16 --epochs 1 

echo meta-llama/Meta-Llama-3-16B-Instruct running
python3 trainig_default.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --train_batch_size 16 \
    --eval_batch_size 16 --epochs 1 
    
echo Qwen/Qwen2-7B-Instruct running
python3 trainig_default.py \
    --model Qwen/Qwen2-7B-Instruct \
    --train_batch_size 16 \
    --eval_batch_size 16 --epochs 1