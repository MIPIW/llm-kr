import argparse
import os
import random
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from typing import Tuple
from trl import SFTTrainer

def get_model_and_tokenizer(args) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_name = args.model_name
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.pad_token
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                    device_map="auto",
                                                    trust_remote_code=True,
                                                    attn_implementation="flash_attention_2",
                                                    torch_dtype=torch.bfloat16)
    hf_model.resize_token_embeddings(len(tokenizer))
    return hf_model, tokenizer

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

def preprocess_function(example):
    final_texts = []
    for i in tqdm(range(len(example['input']))):
        final_text = f"""system

당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.user

아래 질문을 한국어로 정확하게 답변해주세요.
**질문**: {example['input'][i]}assistant\n\n{example['output'][i]}"""
        final_texts.append(final_text)
    return final_texts

def evaluate(args):
    hf_model, tokenizer = get_model_and_tokenizer(args)
    seed_everything(args.random_seed)

    dataset = load_dataset('csv', delimiter=',', data_files={'test': [args.instruction_datasets]}, keep_default_na=False)
    test_dataset = dataset['test']

    training_args = TrainingArguments(
        output_dir='./update/',
        per_device_eval_batch_size=1,
        bf16=True,
    )

    trainer = SFTTrainer(
        hf_model,
        training_args,
        eval_dataset=test_dataset,
        formatting_func=preprocess_function,
        max_seq_length=2048,
        tokenizer=tokenizer,
    )

    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instruction-Tuning Evaluation")
    parser.add_argument("--instruction_datasets", type=str, default="", help="instruction datasets")
    parser.add_argument("--random_seed", type=int, default=42, help="fix random seed in torch, random, numpy")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="model checkpoint path")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="path for evaluation prediction results")

    args = parser.parse_args()

    args.run_name = f"MODEL_NAME:{args.model_name}-DATASETS:{args.instruction_datasets}"
    args.output_dir = f"{args.output_dir}"

    evaluate(args)